import copy
import logging
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import KNNNet
from utils.toolkit import target2onehot, tensor2numpy

from einops import rearrange
from scipy.spatial.distance import cdist

EPSILON = 1e-8

epochs = 1
lrate = 0.1
milestones = [80, 120]
batch_size = 128
weight_decay = 5e-4
num_workers = 8
T = 2


class memknn(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        # memknn require pretrained backbone
        self._network = KNNNet(args["convnet_type"], args['pretrained'], args, self._device)
        self._network.to(self._device)
        self._memory_list = None
        self.k = args['k']

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # with torch.no_grad():
        #     self.build_rehearsal_memory(data_manager, self.samples_per_class)

        if self.args['skip'] and self._cur_task==0:
            load_acc = self._network.load_checkpoint(self.args)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        if self._cur_task == 0:
            if self.args['skip']:
                self._network.to(self._device)
                cur_test_acc = self._compute_accuracy(self._network, self.test_loader)
                logging.info(f"Loaded_Test_Acc:{load_acc} Cur_Test_Acc:{cur_test_acc}")
            else:
                self._train(self.train_loader, self.test_loader) 
                self._compute_accuracy(self._network, self.test_loader)
        else:
            self._train(self.train_loader, self.test_loader)

        #self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        self._memory_list = self._init_memory_list(train_loader)
        self._training_step(train_loader, test_loader)

    def _training_step(self, train_loader, test_loader):
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                out = self._network.convnet(inputs)["features"]
                preds = self._knn(out)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

    def _extract_vectors(self, loader):
        self._network.convnet.eval()
        with torch.no_grad():
            vectors, targets = [], []
            for _, _inputs, _targets in loader:
                _targets = _targets.numpy()
                if isinstance(self._network, nn.DataParallel):
                    _vectors = tensor2numpy(
                        self._network.convnet(_inputs.to(self._device))["features"]
                    )
                else:
                    _vectors = tensor2numpy(
                        self._network.convnet(_inputs.to(self._device))["features"]
                    )

                vectors.append(_vectors)
                targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _reduce_exemplar(self, data_manager, m):
        logging.info("Reducing exemplars...({} per classes)".format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(self._targets_memory)
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = (
                np.concatenate((self._data_memory, dd))
                if len(self._data_memory) != 0
                else dd
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0
                else dt
            )

        if self._memory_list is not None:
            dummy_dms = copy.deepcopy(self._memory_list)
            self._memory_list = dummy_dms[:, :m, :]

    def _construct_exemplar(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)
            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection
                
                if len(vectors) == 0:
                    break

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(selected_exemplars.shape[0], class_idx)

            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )          
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )
            if self._memory_list is None:
                self._memory_list = torch.Tensor(exemplar_vectors).unsqueeze(0).to(self._device)
            else:
                self._memory_list = torch.cat((self._memory_list, torch.Tensor(exemplar_vectors).unsqueeze(0).to(self._device)), dim=0)
        
        self._class_means = self._memory_list.mean(dim=1)
        self._class_means = self._class_means.detach()

        self._memory_list = self._memory_list.detach() # cpu?
        self._class_means = F.normalize(self._class_means, p=2, dim=-1)
        self._memory_list = F.normalize(self._memory_list, p=2, dim=-1)

        self._class_means.requires_grad = False
        self._memory_list.requires_grad = False
   
    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                out = self._network.convnet(inputs)["features"]
                predicts = self._knn(out)
            correct += (predicts == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)
    
    def _knn(self, out):
        with torch.no_grad():
            num_samples = self._memory_list.shape[1]
            all_features = self._memory_list.view([-1, self._memory_list.shape[2]])

            similarity_mat = torch.einsum('b d, n d -> b n', F.normalize(out, dim=-1), F.normalize(all_features, dim=-1))

            topk_sim, indices = similarity_mat.topk(k=self.k, dim=-1, largest=True, sorted=False)

            indices = torch.div(indices, num_samples, rounding_mode='trunc')
            preds, _ = torch.mode(indices)

        return preds

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                out = self._network.convnet(inputs)["features"]
                predicts = self._knn(out)
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = cdist(torch.Tensor.numpy(self._class_means.detach().cpu()), vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]

    def _init_memory_list(self, train_loader):
        max_num_samples = 500
        memory_list = [None] * 100
        count = 0
        with torch.inference_mode():
            for (_, x, y) in tqdm(train_loader):
                x = x.to(self._device)
                out = self._network.convnet(x)['features']
                for out_i, y_i in zip(out, y):
                    if memory_list[y_i] is not None and len(memory_list[y_i]) == max_num_samples:
                        continue

                    out_i = out_i.unsqueeze(0)
                    if memory_list[y_i] is None:
                        memory_list[y_i] = [out_i]
                    else:
                        memory_list[y_i].append(out_i)

        for c in range(100):
            memory_list[c] = torch.cat(memory_list[c], dim=0)

        # num classes, num images, dim
        memory_list = torch.stack(memory_list, dim=0)

        memory_list = memory_list.detach()
        memory_list.requires_grad = False
        return memory_list