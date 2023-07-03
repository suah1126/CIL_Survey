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

from einops import rearrange, repeat
from scipy.spatial.distance import cdist

EPSILON = 1e-8

init_epoch = 1
init_lr = 0.0001
init_milestones = [60, 120]
init_lr_decay = 0.5
init_weight_decay = 0.0005

epochs = 1
lrate = 0.0001
milestones = [60, 120]
lrate_decay = 0.5
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
        self._text_memory_list = None
        self.k = args['k']
        self.model = args['ver']
        self.eval = args['eval']
        self.convnet_type = args['convnet_type']
        self.distillation = args['distillation']
        self.normalize = args['normalize']
        self.epoch = args['epoch']

        if 'clip' in self.convnet_type:
            self.train_mode = 'clip'
            self.test_mode = 'clip'
        else:
            self.train_mode = 'train'
            self.test_mode = 'test'

        print( f"epoch {epochs}\n lrate {lrate}\n milestones {milestones}\n lrate_decay {lrate_decay}\n weight_decay {weight_decay}\n")
        # if args['pretrained']:
        #     init_lr = 0.0001
        #     lrate = 0.0001

    def after_task(self, data_manager):
        # print('|||| modifiy memory ||||')
        # self._construct_exemplar(data_manager, self.samples_per_class, False)
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):

        self._cur_task += 1
        if self._cur_task == 0 and self.model == 'm18':
            self._network.norm_generic_tokens()

        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )

        # image
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode=self.train_mode,
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode=self.test_mode
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # text
        text_dataset = data_manager.get_text_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode=self.train_mode,
            appendent=self._get_text_memory(),
        )
        self.text_loader = DataLoader(
            text_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        with torch.no_grad():
            self.build_rehearsal_memory(data_manager, self.samples_per_class)

        if self.args['skip'] and self._cur_task==0:
            load_acc = self._network.load_checkpoint(self.args)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        if self.eval:
            self._compute_accuracy(self._network, self.test_loader)
        else:
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

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        param_list = []
        for k, v in self._network.named_parameters():
            if 'convnet' in k:
                if not self._network.pretrained:
                    print(k)
                    param_list.append(v)
            else:
                print(k)
                param_list.append(v)

        if self._cur_task == 0:
            optimizer = optim.SGD(
                param_list,
                momentum=0.9,
                lr=init_lr,
                weight_decay=init_weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
            )
            self._training_step(train_loader, test_loader, optimizer, scheduler, True)
        else:
            optimizer = optim.SGD(
                param_list,
                lr=lrate,
                momentum=0.9,
                weight_decay=weight_decay,
            )  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=lrate_decay
            )
            self._training_step(train_loader, test_loader, optimizer, scheduler)

    def _training_step(self, train_loader, test_loader, optimizer, scheduler, init=False):
        '''
        if init:
            prog_bar = tqdm(range(init_epoch))
        else:
            prog_bar = tqdm(range(epochs))
        '''
        prog_bar = tqdm(range(self.epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                logits = self._step(inputs, True)
                loss = F.nll_loss(logits, targets)

                '''
                # distillation loss
                if self.distillation and not init:
                    logits_kd = self._distillation_step(inputs)
                    loss_kd = _KD_loss(
                                    logits[:, : self._known_classes],
                                    logits_kd,
                                    T,)
                    loss = loss + loss_kd
                '''

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

    def _step(self, inputs, training=True):
        out = self._network.convnet(inputs).float() if 'clip' in self.convnet_type else self._network.convnet(inputs)["features"]

        if self.model == 'm8':
            tr_q, tr_knn_cat = self._knn(out, training)
            logits = self._network(tr_q, tr_knn_cat, self._class_means)
            logits = torch.log(logits)
        elif self.model == 'm8_4':
            tr_q, tr_knn_cat = self._knn_classwise(out, training)
            logits = self._network(tr_q, tr_knn_cat, self._class_means)
            logits = torch.log(logits)
        elif self.model == 'm18':
            knnemb = self._knn_18(out, training)
            logits = self._network(out, knnemb, self._class_means, out.shape[0])
            logits = torch.log(logits)
        elif self.model == 'p19_1':
            txtknn, imgknn = self._knn_p19_1(out, training)
            logits = self._network(out, txtknn, imgknn, self._text_class_means, self._class_means, out.shape[0])
            logits = torch.log_softmax(logits, dim=-1)
        elif self.model == 'nakata':
            logits = self._knn_nakata(out, training)
        return logits

    def _distillation_step(self, inputs):
        out_kd = self._old_network.convnet(inputs).float() if 'clip' in self.convnet_type else self._old_network.convnet(inputs)["features"]

        if self.model == 'm8':
            tr_q_kd, tr_knn_cat_kd = self._knn(out_kd, True, True)
            logits_kd = self._old_network(tr_q_kd, tr_knn_cat_kd, self._class_means[:self._known_classes, :])
        elif self.model == 'm8_4':
            tr_q_kd, tr_knn_cat_kd = self._knn_classwise(out_kd, True, True)
            logits_kd = self._old_network(tr_q_kd, tr_knn_cat_kd, self._class_means[:self._known_classes, :])
        elif self.model == 'm18':
            knnemb_kd = self._knn_18(out_kd, True, True)
            logits_kd = self._old_network(out, knnemb_kd, self._class_means[:self._known_classes, :], out_kd.shape[0])
        elif self.model == 'p19_1':
            txtknn_kd, imgknn_kd = self._knn_p19_1(out_kd, training=True, old=True)
            logits_kd = self._old_network(out_kd, txtknn_kd, imgknn_kd, self._text_class_means[:self._known_classes, :], self._class_means[:self._known_classes, :], out_kd.shape[0])

        return logits_kd

    def _extract_vectors(self, loader):
        self._network.convnet.eval()
        with torch.no_grad():
            vectors, targets = [], []
            for _, _inputs, _targets in loader:
                _targets = _targets.numpy()
                if isinstance(self._network, nn.DataParallel):
                    _vectors = tensor2numpy(
                        self._network.convnet(_inputs.to(self._device)).float() if 'clip' in self.convnet_type else self._network.convnet(_inputs.to(self._device))["features"]
                    )
                else:
                    _vectors = tensor2numpy(
                        self._network.convnet(_inputs.to(self._device)).float() if 'clip' in self.convnet_type else self._network.convnet(_inputs.to(self._device))["features"]
                    )

                vectors.append(_vectors)
                targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _extract_text_vectors(self, loader):
        self._network.convnet.eval()
        with torch.no_grad():
            vectors, targets = [], []
            for _, _inputs, _targets in loader:
                _targets = _targets.numpy()
                if isinstance(self._network, nn.DataParallel):
                    _vectors = tensor2numpy(
                        self._network.convnet.encode_text(_inputs.to(self._device)).float()
                    )
                else:
                    _vectors = tensor2numpy(
                        self._network.convnet.encode_text(_inputs.to(self._device)).float()
                    )

                vectors.append(_vectors)
                targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _reduce_exemplar(self, data_manager, m):
        logging.info("Reducing memknn exemplars...({} per classes)".format(m))
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
            # self._memory_list = dummy_dms[:, :m, :]
            self._memory_list = [x[:m] for x in dummy_dms]

    def _reduce_text_exemplar(self, data_manager, m):
        logging.info("Reducing memknn text exemplars...({} per classes)".format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._text_data_memory), copy.deepcopy(self._text_targets_memory)
        self._text_data_memory, self._text_targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._text_data_memory = (
                np.concatenate((self._text_data_memory, dd))
                if len(self._text_data_memory) != 0
                else dd
            )
            self._text_targets_memory = (
                np.concatenate((self._text_targets_memory, dt))
                if len(self._text_targets_memory) != 0
                else dt
            )

        if self._text_memory_list is not None:
            dummy_dms = copy.deepcopy(self._text_memory_list)
            # self._text_memory_list = dummy_dms[:, :m, :]
            self._text_memory_list = [x[:m] for x in dummy_dms]

    def _construct_exemplar(self, data_manager, m, init=True):
        logging.info("Constructing memknn exemplars...({} per classes)".format(m))
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

            if init:
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
            else:
                idx = m*class_idx
                self._data_memory[idx:idx+m] = selected_exemplars
                self._targets_memory[idx:idx+m] = exemplar_targets

            if self._memory_list is None:
                self._memory_list = [torch.Tensor(exemplar_vectors).to(self._device)]
            elif init:
                # self._memory_list = torch.cat((self._memory_list, torch.Tensor(exemplar_vectors).unsqueeze(0).to(self._device)), dim=0)
                self._memory_list.append(torch.Tensor(exemplar_vectors).to(self._device))
            else:
                assert False, "let's don't assume we modify img memory online"
                # self._memory_list[class_idx, :, :] = torch.Tensor(exemplar_vectors).unsqueeze(0).to(self._device)
        # self._class_means = self._memory_list.mean(dim=1)
        self._class_means = torch.stack([x.mean(dim=0) for x in self._memory_list], dim=0)
        self._class_means = self._class_means.detach()

        if self.normalize:
            self._class_means = F.normalize(self._class_means, p=2, dim=-1)
        #self._memory_list = F.normalize(self._memory_list, p=2, dim=-1)

        self._class_means.requires_grad = False

    def _construct_text_exemplar(self, data_manager, m, init=True):
        logging.info("Constructing memknn text exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_text_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_text_vectors(idx_loader)
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

            if init:
                self._text_data_memory = (
                    np.concatenate((self._text_data_memory, selected_exemplars))
                    if len(self._text_data_memory) != 0
                    else selected_exemplars
                )
                self._text_targets_memory = (
                    np.concatenate((self._text_targets_memory, exemplar_targets))
                    if len(self._text_targets_memory) != 0
                    else exemplar_targets
                )
            else:
                idx = m * class_idx
                self._text_data_memory[idx:idx+m] = selected_exemplars
                self._text_targets_memory[idx:idx+m] = exemplar_targets

            if self._text_memory_list is None:
                self._text_memory_list = [torch.Tensor(exemplar_vectors).to(self._device)]
            elif init:
                # self._text_memory_list = torch.cat((self._text_memory_list, torch.Tensor(exemplar_vectors).unsqueeze(0).to(self._device)), dim=0)
                # self._text_memory_list = torch.cat((self._text_memory_list, torch.Tensor(exemplar_vectors).to(self._device)), dim=0)
                self._text_memory_list.append(torch.Tensor(exemplar_vectors).to(self._device))
            else:
                assert False, "let's don't assume we modify text memory online"
                # self._text_memory_list[class_idx, :, :] = torch.Tensor(exemplar_vectors).unsqueeze(0).to(self._device)
        # self._text_class_means = self._text_memory_list.mean(dim=1)
        self._text_class_means = torch.stack([x.mean(dim=0) for x in self._text_memory_list], dim=0)
        self._text_class_means = self._text_class_means.detach()

        if self.normalize:
            self._text_class_means = F.normalize(self._text_class_means, p=2, dim=-1)
        #self._memory_list = F.normalize(self._memory_list, p=2, dim=-1)

        self._text_class_means.requires_grad = False

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._step(inputs, False)
            predicts = torch.argmax(logits, dim=1)
            correct += (predicts == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _knn(self, out, training, old=False):
        with torch.no_grad():
            if old:
                classwise_sim = torch.einsum('b d, n d -> b n', out, rearrange(self._memory_list[:self._known_classes, :, :], 'c n d -> (c n) d'))
            else:
                classwise_sim = torch.einsum('b d, n d -> b n', out, rearrange(self._memory_list, 'c n d -> (c n) d'))

            # B, N -> B, K
            if training:
                topk_sim, indices = classwise_sim.topk(k=self.k + 1, dim=-1, largest=True, sorted=True)
                indices = indices[:, 1:]
            else:
                topk_sim, indices = classwise_sim.topk(k=self.k, dim=-1, largest=True, sorted=True)

            # C, N, D [[B, K]] -> B, K, D
            knnemb = rearrange(self._memory_list, 'c n d -> (c n) d')[indices]

            # corresponding_proto = self.global_proto[class_ids]  # self.global_proto_learned(class_ids)
            # B, 1, D
            tr_q = out.unsqueeze(1)
            # (B, 1, D), (B, C, D) -> B, (1 + C), D
            tr_knn_cat = torch.cat([tr_q, knnemb], dim=1)

        return out.unsqueeze(1), tr_knn_cat

    def _knn_classwise(self, out, training, old=False):
        batchsize = out.shape[0]
        with torch.no_grad():
            if old:
                classwise_sim = torch.einsum('b d, c n d -> b c n', out, self._memory_list[:self._known_classes, :, :])
                class_idx = torch.arange(self._known_classes).unsqueeze(0).unsqueeze(-1)
            else:
                classwise_sim = torch.einsum('b d, c n d -> b c n', out, self._memory_list)
                class_idx = torch.arange(self._total_classes).unsqueeze(0).unsqueeze(-1)

            # B, N -> B, K
            if training:  # to ignore self-voting
                # B, C, N -> B, C, K
                topk_sim, indices = classwise_sim.topk(k=self.k + 1, dim=-1, largest=True, sorted=True)
                top1_sim = topk_sim[:, :, 0]
                max_class_indices = top1_sim.argmax(dim=1)  # highly likely the self (or another twin in the feature space)
                indices[range(batchsize), max_class_indices, :-1] = indices[range(batchsize), max_class_indices, 1:]
                indices = indices[:, :, :-1]
            else:
                topk_sim, indices = classwise_sim.topk(k=self.k, dim=-1, largest=True, sorted=True)

            # C, N, D [[1, C, 1], [B, C, K]] -> B, C, K, D
            knnemb = self._memory_list[class_idx, indices]
            knnemb = rearrange(knnemb, 'b c k d -> b (c k) d')
            # B, 1, D
            tr_q = out.unsqueeze(1)
            # (B, 1, D), (B, C, D) -> B, (1 + C), D
            tr_knn_cat = torch.cat([tr_q, knnemb], dim=1)

        return out.unsqueeze(1), tr_knn_cat

    def _knn_18(self, out, training, old=False):
        bs = out.shape[0]
        #out = F.normalize(out, p=2, dim=-1)
        with torch.no_grad():
            classwise_sim = torch.einsum('b d, c n d -> b c n', out, self._memory_list)
            if self._network.convnet.training:  # to ignore self-voting
                # B, C, N -> B, C, K
                topk_sim, indices = classwise_sim.topk(k=self.k + 1, dim=-1, largest=True, sorted=True)
                # indices = indices[:, :, 1:]
                top1_indices = indices[:, :, 0]
                max_class_indices = top1_indices.argmax(dim=1)  # highly likely the self (or another twin in the feature space)
                indices[range(bs), max_class_indices, :-1] = indices[range(bs), max_class_indices, 1:]
                indices = indices[:, :, :-1]
            else:
                topk_sim, indices = classwise_sim.topk(k=self.k, dim=-1, largest=True, sorted=True)
            # 1, C, 1
            class_idx = torch.arange(self._total_classes).unsqueeze(0).unsqueeze(-1)
            # C, N, D [[1, C, 1], [B, C, K]] -> B, C, K, D
            knnemb = self._memory_list[class_idx, indices]
            knnemb = torch.cat([repeat(self._class_means, 'c d -> b c 1 d', b=bs), knnemb], dim=2)
            # knnemb = rearrange(knnemb, 'b c k d -> (b c) k d')
            knnemb = rearrange(knnemb, 'b c k d -> b (c k) d')

        return knnemb

    def _knn_p19_1(self, out, training, old=False):
        def retrieve_knn(x, mem, k):
            with torch.no_grad():
                classwise_sim = torch.einsum('b d, n d -> b n', x, mem)
                _, indices = classwise_sim.topk(k=k, dim=-1, largest=True, sorted=True)

                # N, D [[B, K] -> B, K, D
                knnemb = mem[indices]
                return knnemb

        if old:
            mem_txt = torch.cat(self._text_memory_list[:self._known_classes], dim=0)
            mem_img = torch.cat(self._memory_list[:self._known_classes], dim=0)
        else:
            mem_txt = torch.cat(self._text_memory_list, dim=0)
            mem_img = torch.cat(self._memory_list, dim=0)

        kv_txt = retrieve_knn(x=out, mem=mem_txt, k=self.k)
        kv_img = retrieve_knn(x=out, mem=mem_img, k=self.k)
        return kv_txt, kv_img
        # return kv_img, kv_img  # 84.0

    def _knn_nakata(self, out, training):

        labels  = torch.tensor([[i] * len(mem) for i, mem in enumerate(self._memory_list)]).flatten().to(out.device)
        memory = torch.cat(self._memory_list, dim=0).to(out.device)
        num_cls = max(labels)+1

        # l2_norm
        out_ = F.normalize(out, dim=-1, p=2)
        memory_ = F.normalize(memory, dim=-1, p=2)

        globalsim = torch.einsum('b d, n d -> b n', out_, memory_)
        _, indices = globalsim.topk(k=self.k, dim=-1, largest=True, sorted=True)
        globalcls = labels[indices]

        def majority_vote(input):
            count = [0]*num_cls
            for item in input:
                count[item.cpu().item()] += 1

            onehot = torch.argmax(torch.tensor(count)).item() # just vote
            result = torch.tensor([0.]*num_cls)
            result[onehot] = 1.0

            return result.to(input.device)

        sim = torch.stack(list(map(majority_vote, globalcls)))

        return sim

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._step(inputs, False)
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
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

def _KD_loss(pred, soft, T):
    # pred = torch.log_softmax(pred / T, dim=1)
    # soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
