import copy
import logging
import torch
from torch import nn
from convs.cifar_resnet import resnet32
from convs.resnet import resnet18, resnet34, resnet50
from convs.ucir_cifar_resnet import resnet32 as cosine_resnet32
from convs.ucir_resnet import resnet18 as cosine_resnet18
from convs.ucir_resnet import resnet34 as cosine_resnet34
from convs.ucir_resnet import resnet50 as cosine_resnet50
from convs.linears import SimpleLinear, SplitCosineLinear, CosineLinear

# FOR MEMO
from convs.memo_resnet import  get_resnet18_imagenet as get_memo_resnet18 #for imagenet
from convs.memo_cifar_resnet import get_resnet32_a2fc as get_memo_resnet32 #for cifar

# FOR AUC & DER
from convs.conv_cifar import conv2 as conv2_cifar
from convs.cifar_resnet import resnet14 as resnet14_cifar
from convs.cifar_resnet import resnet20 as resnet20_cifar
from convs.cifar_resnet import resnet26 as resnet26_cifar

from convs.conv_imagenet import conv4 as conv4_imagenet
from convs.resnet import resnet10 as resnet10_imagenet
from convs.resnet import resnet26 as resnet26_imagenet
from convs.resnet import resnet34 as resnet34_imagenet
from convs.resnet import resnet50 as resnet50_imagenet

# FOR AUC & MEMO
from convs.conv_cifar import get_conv_a2fc as memo_conv2_cifar
from convs.memo_cifar_resnet import get_resnet14_a2fc as memo_resnet14_cifar
from convs.memo_cifar_resnet import get_resnet20_a2fc as memo_resnet20_cifar
from convs.memo_cifar_resnet import get_resnet26_a2fc as memo_resnet26_cifar

from convs.conv_imagenet import conv_a2fc_imagenet as memo_conv4_imagenet
from convs.memo_resnet import get_resnet10_imagenet as memo_resnet10_imagenet
from convs.memo_resnet import get_resnet26_imagenet as memo_resnet26_imagenet
from convs.memo_resnet import get_resnet34_imagenet as memo_resnet34_imagenet
from convs.memo_resnet import get_resnet50_imagenet as memo_resnet50_imagenet

from utils.transformer import TransformerEncoderLayer
from models.transformer import ResidualAttentionBlock

import torchvision
import torch.nn.functional as F

from einops import rearrange, repeat
import clip

def get_convnet(convnet_type, pretrained=False):
    name = convnet_type.lower()
    if name == "resnet32":
        return resnet32()
    elif name == "resnet18":
        return resnet18(pretrained=pretrained)
    elif name == "resnet34":
        return resnet34(pretrained=pretrained)
    elif name == "resnet50":
        return resnet50(pretrained=pretrained)
    elif name == "cosine_resnet18":
        return cosine_resnet18(pretrained=pretrained)
    elif name == "cosine_resnet32":
        return cosine_resnet32()
    elif name == "cosine_resnet34":
        return cosine_resnet34(pretrained=pretrained)
    elif name == "cosine_resnet50":
        return cosine_resnet50(pretrained=pretrained)

    # MEMO benchmark backbone
    elif name == 'memo_resnet18':
        _basenet, _adaptive_net = get_memo_resnet18()
        return _basenet, _adaptive_net
    elif name == 'memo_resnet32':
        _basenet, _adaptive_net = get_memo_resnet32()
        return _basenet, _adaptive_net

    # AUC
    ## cifar
    elif name == 'conv2':
        return conv2_cifar()
    elif name == 'resnet14_cifar':
        return resnet14_cifar()
    elif name == 'resnet20_cifar':
        return resnet20_cifar()
    elif name == 'resnet26_cifar':
        return resnet26_cifar()

    elif name == 'memo_conv2':
        g_blocks, s_blocks = memo_conv2_cifar() # generalized/specialized
        return g_blocks, s_blocks
    elif name == 'memo_resnet14_cifar':
        g_blocks, s_blocks = memo_resnet14_cifar() # generalized/specialized
        return g_blocks, s_blocks
    elif name == 'memo_resnet20_cifar':
        g_blocks, s_blocks = memo_resnet20_cifar() # generalized/specialized
        return g_blocks, s_blocks
    elif name == 'memo_resnet26_cifar':
        g_blocks, s_blocks = memo_resnet26_cifar() # generalized/specialized
        return g_blocks, s_blocks

    ## imagenet
    elif name == 'conv4':
        return conv4_imagenet()
    elif name == 'resnet10_imagenet':
        return resnet10_imagenet()
    elif name == 'resnet26_imagenet':
        return resnet26_imagenet()
    elif name == 'resnet34_imagenet':
        return resnet34_imagenet()
    elif name == 'resnet50_imagenet':
        return resnet50_imagenet()

    elif name == 'memo_conv4':
        g_blcoks, s_blocks = memo_conv4_imagenet()
        return g_blcoks, s_blocks
    elif name == 'memo_resnet10_imagenet':
        g_blcoks, s_blocks = memo_resnet10_imagenet()
        return g_blcoks, s_blocks
    elif name == 'memo_resnet26_imagenet':
        g_blcoks, s_blocks = memo_resnet26_imagenet()
        return g_blcoks, s_blocks
    elif name == 'memo_resnet34_imagenet':
        g_blocks, s_blocks = memo_resnet34_imagenet()
        return g_blocks, s_blocks
    elif name == 'memo_resnet50_imagenet':
        g_blcoks, s_blocks = memo_resnet50_imagenet()
        return g_blcoks, s_blocks

    elif name == 'clipvitb':
        model, _ = clip.load("ViT-B/32")
        model.forward = model.encode_image
        return model
    else:
        raise NotImplementedError("Unknown type {}".format(convnet_type))


class BaseNet(nn.Module):
    def __init__(self, convnet_type, pretrained):
        super(BaseNet, self).__init__()

        #self.feature_dim = 2048 if convnet_type == 'resnet50' else 512
        self.convnet_type = convnet_type
        self.convnet = get_convnet(convnet_type, pretrained)
        self.fc = None

    @property
    def feature_dim(self):
        return 512 if 'clip' in self.convnet_type else self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)["features"]

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update(x)

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

    def load_checkpoint(self, args):
        if args["init_cls"] == 50:
            pkl_name = "{}_{}_{}_B{}_Inc{}".format(
                args["dataset"],
                args["seed"],
                args["convnet_type"],
                0,
                args["init_cls"],
            )
            checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        self.convnet.load_state_dict(model_infos['convnet'])
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc

class IncrementalNet(BaseNet):
    def __init__(self, convnet_type, pretrained, gradcam=False):
        super().__init__(convnet_type, pretrained)
        self.gradcam = gradcam
        self.convnet_type = convnet_type
        if hasattr(self, "gradcam") and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

        # if pretrained:
        #     self.convnet.requires_grad_(requires_grad=False)

    def extract_vector(self, x):
        return self.convnet(x).float() if 'clip' in self.convnet_type else self.convnet(x)["features"]

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def forward(self, x):
        x = self.convnet(x)
        if 'clip' in self.convnet_type:
            out = self.fc(x.float())
        else:
            out = self.fc(x["features"])
            out.update(x)
        if hasattr(self, "gradcam") and self.gradcam:
            out["gradcam_gradients"] = self._gradcam_gradients
            out["gradcam_activations"] = self._gradcam_activations

        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(
            backward_hook
        )
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(
            forward_hook
        )


class CosineIncrementalNet(BaseNet):
    def __init__(self, convnet_type, pretrained, nb_proxy=1):
        super().__init__(convnet_type, pretrained)
        self.nb_proxy = nb_proxy

    def update_fc(self, nb_classes, task_num):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            if task_num == 1:
                fc.fc1.weight.data = self.fc.weight.data
                fc.sigma.data = self.fc.sigma.data
            else:
                prev_out_features1 = self.fc.fc1.out_features
                fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
                fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
                fc.sigma.data = self.fc.sigma.data

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        if self.fc is None:
            fc = CosineLinear(in_dim, out_dim, self.nb_proxy, to_reduce=True)
        else:
            prev_out_features = self.fc.out_features // self.nb_proxy
            # prev_out_features = self.fc.out_features
            fc = SplitCosineLinear(
                in_dim, prev_out_features, out_dim - prev_out_features, self.nb_proxy
            )

        return fc


class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x, low_range, high_range):
        ret_x = x.clone()
        ret_x[:, low_range:high_range] = (
            self.alpha * x[:, low_range:high_range] + self.beta
        )
        return ret_x

    def get_params(self):
        return (self.alpha.item(), self.beta.item())


class IncrementalNetWithBias(BaseNet):
    def __init__(self, convnet_type, pretrained, bias_correction=False):
        super().__init__(convnet_type, pretrained)

        # Bias layer
        self.bias_correction = bias_correction
        self.bias_layers = nn.ModuleList([])
        self.task_sizes = []

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        if self.bias_correction:
            logits = out["logits"]
            for i, layer in enumerate(self.bias_layers):
                logits = layer(
                    logits, sum(self.task_sizes[:i]), sum(self.task_sizes[: i + 1])
                )
            out["logits"] = logits

        out.update(x)

        return out

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.bias_layers.append(BiasLayer())

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def get_bias_params(self):
        params = []
        for layer in self.bias_layers:
            params.append(layer.get_params())

        return params

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


class DERNet(nn.Module):
    def __init__(self, convnet_type, pretrained):
        super(DERNet, self).__init__()
        self.convnet_type = convnet_type
        self.convnets = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.aux_fc = None
        self.task_sizes = []

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)

        out = self.fc(features)  # {logics: self.fc(features)}

        aux_logits = self.aux_fc(features[:, -self.out_dim :])["logits"]

        out.update({"aux_logits": aux_logits, "features": features})
        return out
        """
        {
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }
        """

    def update_fc(self, nb_classes):
        if len(self.convnets) == 0:
            self.convnets.append(get_convnet(self.convnet_type))
        else:
            self.convnets.append(get_convnet(self.convnet_type))
            self.convnets[-1].load_state_dict(self.convnets[-2].state_dict())

        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)

        self.aux_fc = self.generate_fc(self.out_dim, new_task_size + 1)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def load_checkpoint(self, args):
        checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        assert len(self.convnets) == 1
        self.convnets[0].load_state_dict(model_infos['convnet'])
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc


class SimpleCosineIncrementalNet(BaseNet):
    def __init__(self, convnet_type, pretrained):
        super().__init__(convnet_type, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:

                weight = torch.cat([weight, nextperiod_initialization])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc


class FOSTERNet(nn.Module):
    def __init__(self, convnet_type, pretrained):
        super(FOSTERNet, self).__init__()
        self.convnet_type = convnet_type
        self.convnets = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.fe_fc = None
        self.task_sizes = []
        self.oldfc = None

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        out = self.fc(features)
        fe_logits = self.fe_fc(features[:, -self.out_dim :])["logits"]

        out.update({"fe_logits": fe_logits, "features": features})

        if self.oldfc is not None:
            old_logits = self.oldfc(features[:, : -self.out_dim])["logits"]
            out.update({"old_logits": old_logits})

        out.update({"eval_logits": out["logits"]})
        return out

    def update_fc(self, nb_classes):
        self.convnets.append(get_convnet(self.convnet_type))
        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias
            self.convnets[-1].load_state_dict(self.convnets[-2].state_dict())

        self.oldfc = self.fc
        self.fc = fc
        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.fe_fc = self.generate_fc(self.out_dim, nb_classes)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def copy_fc(self, fc):
        weight = copy.deepcopy(fc.weight.data)
        bias = copy.deepcopy(fc.bias.data)
        n, m = weight.shape[0], weight.shape[1]
        self.fc.weight.data[:n, :m] = weight
        self.fc.bias.data[:n] = bias

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, old, increment, value):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew * (value ** (old / increment))
        logging.info("align weights, gamma = {} ".format(gamma))
        self.fc.weight.data[-increment:, :] *= gamma


    def load_checkpoint(self, args):
        if args["init_cls"] == 50:
            pkl_name = "{}_{}_{}_B{}_Inc{}".format(
                args["dataset"],
                args["seed"],
                args["convnet_type"],
                0,
                args["init_cls"],
            )
            checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        assert len(self.convnets) == 1
        self.convnets[0].load_state_dict(model_infos['convnet'])
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc

class AdaptiveNet(nn.Module):
    def __init__(self, convnet_type, pretrained):
        super(AdaptiveNet, self).__init__()
        self.convnet_type=convnet_type
        self.TaskAgnosticExtractor , _ = get_convnet(convnet_type, pretrained) #Generalized blocks
        self.TaskAgnosticExtractor.train()
        self.AdaptiveExtractors = nn.ModuleList() #Specialized Blocks
        self.pretrained=pretrained
        self.out_dim=None
        self.fc = None
        self.aux_fc=None
        self.task_sizes = []

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim*len(self.AdaptiveExtractors)

    def extract_vector(self, x):
        base_feature_map = self.TaskAgnosticExtractor(x)
        features = [extractor(base_feature_map) for extractor in self.AdaptiveExtractors]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        base_feature_map = self.TaskAgnosticExtractor(x)
        features = [extractor(base_feature_map) for extractor in self.AdaptiveExtractors]
        features = torch.cat(features, 1)
        out=self.fc(features) #{logits: self.fc(features)}

        aux_logits=self.aux_fc(features[:,-self.out_dim:])["logits"]

        out.update({"aux_logits":aux_logits,"features":features})
        out.update({"base_features":base_feature_map})
        return out

        '''
        {
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }
        '''

    def update_fc(self,nb_classes):
        _ , _new_extractor = get_convnet(self.convnet_type)
        if len(self.AdaptiveExtractors)==0:
            self.AdaptiveExtractors.append(_new_extractor)
        else:
            self.AdaptiveExtractors.append(_new_extractor)
            self.AdaptiveExtractors[-1].load_state_dict(self.AdaptiveExtractors[-2].state_dict())

        if self.out_dim is None:
            logging.info(self.AdaptiveExtractors[-1])
            self.out_dim=self.AdaptiveExtractors[-1].feature_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output,:self.feature_dim-self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.aux_fc=self.generate_fc(self.out_dim,new_task_size+1)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def weight_align(self, increment):
        weights=self.fc.weight.data
        newnorm=(torch.norm(weights[-increment:,:],p=2,dim=1))
        oldnorm=(torch.norm(weights[:-increment,:],p=2,dim=1))
        meannew=torch.mean(newnorm)
        meanold=torch.mean(oldnorm)
        gamma=meanold/meannew
        print('alignweights,gamma=',gamma)
        self.fc.weight.data[-increment:,:]*=gamma

    def load_checkpoint(self, args):
        if args["init_cls"] == 50:
            pkl_name = "{}_{}_{}_B{}_Inc{}".format(
                args["dataset"],
                args["seed"],
                args["convnet_type"],
                0,
                args["init_cls"],
            )
            checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        checkpoint_name = checkpoint_name.replace("memo_", "")
        model_infos = torch.load(checkpoint_name)
        model_dict = model_infos['convnet']
        assert len(self.AdaptiveExtractors) == 1

        base_state_dict = self.TaskAgnosticExtractor.state_dict()
        adap_state_dict = self.AdaptiveExtractors[0].state_dict()

        pretrained_base_dict = {
            k:v
            for k, v in model_dict.items()
            if k in base_state_dict
        }

        pretrained_adap_dict = {
            k:v
            for k, v in model_dict.items()
            if k in adap_state_dict
        }

        base_state_dict.update(pretrained_base_dict)
        adap_state_dict.update(pretrained_adap_dict)

        self.TaskAgnosticExtractor.load_state_dict(base_state_dict)
        self.AdaptiveExtractors[0].load_state_dict(adap_state_dict)
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc

class KNNNet(BaseNet):
    def __init__(self, convnet_type, pretrained, args, device, gradcam=False):
        super().__init__(convnet_type, pretrained)
        self.gradcam = gradcam
        self.pretrained = pretrained
        if hasattr(self, "gradcam") and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

        self.args = args
        if self.pretrained:
        #     #self.convnet = torchvision.models.resnet18(pretrained=True)
            self.convnet.requires_grad_(requires_grad=False)
        self.convnet.fc = nn.Identity()

        self.dim = self.feature_dim
        if self.args['ver'] == 'p19_1':
            self.attn_txt = ResidualAttentionBlock(d_model=self.dim, n_head=1)
            self.attn_img = ResidualAttentionBlock(d_model=self.dim, n_head=1)
        else:
            self.nhead = 8
            self._dtype = torch.float32
            self.knnformer2 = TransformerEncoderLayer(d_model=self.dim,
                                                    nhead=self.nhead,
                                                    dim_feedforward=self.dim,
                                                    dropout=0.0,
                                                    # activation=F.relu,
                                                    layer_norm_eps=1e-05,
                                                    batch_first=True,
                                                    norm_first=True,
                                                    device=device,
                                                    dtype=self._dtype,
                                                    )
            self.knnformer = TransformerEncoderLayer(d_model=self.dim,
                                                    nhead=self.nhead,
                                                    dim_feedforward=self.dim,
                                                    dropout=0.0,
                                                    # activation=F.relu,
                                                    layer_norm_eps=1e-05,
                                                    batch_first=True,
                                                    norm_first=True,
                                                    device=device,
                                                    dtype=self._dtype,
                                                    )

        if 'm8' in self.args['ver']:
            self.forward = self.forward_m8
        elif self.args['ver'] == 'm18':
            self.forward = self.forward_m18
            self.generic_tokens = self._init_generic_tokens()
        elif self.args['ver'] == 'nakata':
            self.forward = self.forward_nakata
        elif self.args['ver'] == 'p19_1':
            self.forward = self.forward_p19_1

    def _init_generic_tokens(self):
        _generic_tokens = torch.empty(self.args['ntokens'], self.dim, dtype=self._dtype, requires_grad=True)
        generic_tokens = nn.Parameter(_generic_tokens.clone(), requires_grad=True)
        # moved to self.on_fit_start; should be called after params being loaded to cuda
        # nn.init.trunc_normal_(self.generic_tokens, mean=0.0, std=0.02)
        return generic_tokens

    # def extract_vector(self, x):
    #     return self.convnet(x).float()

    def norm_generic_tokens(self):
        self.generic_tokens = nn.init.trunc_normal_(self.generic_tokens, mean=0.0, std=0.02)

    def forward_m18(self, out, knnemb, global_proto, batchsize):
        updated_tokens = self.knnformer(repeat(self.generic_tokens, 'm d -> b m d', b=batchsize), knnemb, knnemb)
        updated_tokens = F.normalize(updated_tokens, p=2, eps=1e-6, dim=-1)

        # no kNN baseline; no prototype update at all!
        # gtokens = self.generic_tokens.unsqueeze(0)
        # _updated_proto = self.knnformer2(self.global_proto.unsqueeze(0), gtokens, gtokens)

        # (B, C, D), (B, M, D) -> B, C, D
        _updated_proto = self.knnformer2(repeat(global_proto, 'c d -> b c d', b=batchsize), updated_tokens, updated_tokens)

        # standization & l2 normalization
        _updated_proto = self.standardize(_updated_proto)

        # output becomes NaN if it's commented!!
        updated_proto = F.normalize(_updated_proto, p=2, eps=1e-6, dim=-1)

        sim = torch.einsum('b d, b c d -> b c', out, updated_proto)

        # no kNN baseline; no prototype update at all!
        # sim = torch.einsum('b d, c d -> b c', out, updated_proto.squeeze(0))
        return F.softmax(sim, dim=1)

    def forward_m8(self, tr_q, tr_knn_cat, global_proto, T=1):
        qout = self.knnformer2(tr_q, tr_q, tr_q)
        nout = self.knnformer(tr_q, tr_knn_cat, tr_knn_cat)

        qout = torch.einsum('b d, c d -> b c', qout[:, 0], global_proto)
        nout = torch.einsum('b d, c d -> b c', nout[:, 0], global_proto)

        # return torch.log(0.5 * (F.softmax(qout, dim=1) + F.softmax(nout, dim=1)))
        avgprob = 0.5 * (F.softmax(qout / T, dim=1) + F.softmax(nout / T, dim=1))
        avgprob = torch.clamp(avgprob, 1e-6)  # to prevent numerical unstability
        return avgprob

    def forward_nakata(self, x):
        return self.convnet(x)['features']

    def standardize(self, x, dim=1, eps=1e-6):
        out = x - x.mean(dim=dim, keepdim=True)
        out = out / (out.std(dim=dim, keepdim=True) + eps)
        return out

    def forward_p19_1(self, clipfeat, kv_txt, kv_img, txt_proto, img_proto, dim):
        out_txt = self.attn_txt(clipfeat.unsqueeze(1), kv_txt, kv_txt).squeeze(1)
        out_img = self.attn_img(clipfeat.unsqueeze(1), kv_img, kv_img).squeeze(1)

        out_txt_ = F.normalize(out_txt, dim=-1, p=2)
        out_img_ = F.normalize(out_img, dim=-1, p=2)
        clipfeat_ = F.normalize(clipfeat, dim=-1, p=2)
        proto_txt_ = F.normalize(txt_proto.to(clipfeat.device), dim=-1, p=2)
        proto_img_ = F.normalize(img_proto.to(clipfeat.device), dim=-1, p=2)

        base_scale = self.args['base_scale']
        knn_scale = self.args['knn_scale']
        print(base_scale, knn_scale)
        sim_clip = torch.einsum('c d, b d -> b c', proto_txt_, clipfeat_) * base_scale
        sim_txt = torch.einsum('c d, b d -> b c', proto_img_, out_txt_) * knn_scale
        sim_img = torch.einsum('c d, b d -> b c', proto_txt_, out_img_) * knn_scale

        sim = sim_clip + sim_txt + sim_img

        return sim

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(
            backward_hook
        )
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(
            forward_hook
        )
