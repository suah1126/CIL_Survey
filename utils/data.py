import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from utils.toolkit import split_images_labels
from torchvision.transforms import InterpolationMode
from PIL import Image
from utils.text.preprocess import SentPreProcessor

BICUBIC = Image.BICUBIC
dataroot = '/home/dahyun/datasets'

def convert_image_to_rgb(image):
    return image.convert("RGB")

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10(dataroot, train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10(dataroot, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    img_size = 224
    train_trsf = [
        #transforms.Resize((img_size, img_size)),  # TODO: randomcrop?
        # transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize((32, 32)),
    ]
    common_trsf = [
        transforms.ToTensor(),
        #transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]
    clip_trsf = [
            transforms.Resize(img_size, interpolation=BICUBIC),
            transforms.CenterCrop(img_size),
            convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100(dataroot, train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100(dataroot, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    clip_trsf = [
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    class_order = np.arange(1000).tolist()

    def download_data(self):
        train_dir = os.path.join(dataroot, "imagenet100/train/")
        test_dir = os.path.join(dataroot, "imagenet100/val/")

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

        # build text class data
        # build classname-classid key: {'n01234567': 0, ... }
        self.loaded_idxs = {}
        for x, y in zip(self.test_data, self.test_targets):
            classname = x.split('/')[-2]  # n01234567
            if self.loaded_idxs.get(classname) == None:
                self.loaded_idxs[classname] = y

        # sentence token generator
        label_mapping_file = os.path.join("imagenet100", 'labels.txt')
        wiki_dir = os.path.join("imagenet100", 'wiki')
        self.text_tokens = self._make_sentence_tokens(label_mapping_file, wiki_dir)
        self.num_sents = [token.shape[0] for token in self.text_tokens]
        self.text = TextToken_Dataset(self.text_tokens, self.num_sents)
        # self.text_data, self.text_targets = self.text.data, self.text.targets

    def _make_sentence_tokens(self, label_mapping_file, wiki_dir):
        preprocessor = SentPreProcessor(dataroot, self.loaded_idxs, label_mapping_file, wiki_dir, context_length=75)
        return preprocessor.make_sentence_tokens()


class TextToken_Dataset(Dataset):
    def __init__(self, text_tokens: list, num_sents: list):
        self.data = torch.cat(text_tokens)
        self.targets = []

        targets = [[idx]*nsents for idx, nsents in enumerate(num_sents)]
        for t in targets: self.targets += t

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.targets[index]

        return sample, label
