import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import KNNNet
from utils.toolkit import target2onehot, tensor2numpy

EPSILON = 1e-8

epochs = 170
lrate = 5e-3
milestones = [80, 120]
batch_size = 128
weight_decay = 5e-4
num_workers = 8
T = 2


class memknn(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        # memknn require pretrained backbone
        self._network = KNNNet(args["convnet_type"], True, args, self._device)
        self.mode = 'embedding'
        self.k = args['k']

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        # self._network.update_fc(self._total_classes)
        # logging.info(
        #     "Learning on {}-{}".format(self._known_classes, self._total_classes)
        # )

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

        self.build_rehearsal_memory(data_manager, self.samples_per_class, self.mode)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        param_list = []
        for k, v in self._network.named_parameters():
            if not 'backbone' in k:
                param_list.append(v)
            else:
                print(k)
        optimizer = optim.SGD(
            param_list,
            # list(self.knnformer.parameters()) + list(self.fc.parameters()),
            lr=lrate,
            momentum=0.9,
            weight_decay=weight_decay,
        )
        self._training_step(train_loader, test_loader, optimizer)

    def _training_step(self, train_loader, test_loader, optimizer):
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.module.qinformer.train()
            self._network.module.knnformer.train()
            self._network.module.convnet.eval()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                
                out = self._network.module.convnet(inputs)
                print(out)
                with torch.no_grad():
                    classwise_sim = torch.einsum('b d, n d -> b n', out, self._data_memory)
                    # B, N -> B, K
                    topk_sim, indices = classwise_sim.topk(k=self.k, dim=-1, largest=True, sorted=False)

                    # C, N, D [[B, K]] -> B, K, D
                    knnemb = self._data_memory[indices]

                    # corresponding_proto = self.global_proto[class_ids]  # self.global_proto_learned(class_ids)
                    # B, 1, D
                    tr_q = out.unsqueeze(1)
                    # (B, 1, D), (B, C, D) -> B, (1 + C), D
                    tr_knn_cat = torch.cat([tr_q, knnemb], dim=1)

                logits = self._network(inputs, tr_q, tr_knn_cat, self._class_means)
                loss = F.nll_loss(logits, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
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
        self._network.module.convnet.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(
                    self._network.module.convnet(_inputs.to(self._device))
                )
            else:
                _vectors = tensor2numpy(
                    self._network.convnet(_inputs.to(self._device))
                )

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)