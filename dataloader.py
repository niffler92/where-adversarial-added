import os
from PIL import Image

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from common.torch_utils import one_hot
from settings import PROJECT_ROOT
from nsml import HAS_DATASET, DATASET_PATH, NSML_NFS_OUTPUT

__all__ = ['MNIST', 'CIFAR10', 'CIFAR100', 'ImageNet', 'TinyImageNet', 'COCO']


def data_stats(dataset):
    """
    Returns: *mean, *std
    """
    if dataset in ["CIFAR10", "CIFAR100", "TinyImageNet", "ImageNet", "COCO"]:
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif dataset == "MNIST":
        return [0.1307], [0.3081]


def denormalize(images, dataset, clamp=True):
    """
    Args:
        images (Variable) - (B, C, H, W)
        dataset (str)

    Returns normalized -> denormalized to [0, 1] scale
    """
    mean, std = data_stats(dataset)
    mean, std = torch.FloatTensor(mean), torch.FloatTensor(std)
    if torch.cuda.is_available():
        mean, std = mean.cuda(), std.cuda()
    mean, std = mean.view(1,-1,1,1), std.view(1,-1,1,1)

    if clamp:
        return torch.clamp((images*std + mean), 0, 1)
    else:
        return images*std + mean


def normalize(images, dataset):
    """
    Args:
        images (Variable) - (B, C, H, W)
        dataset (str)
    """
    mean, std = data_stats(dataset)
    mean, std = torch.FloatTensor(mean), torch.FloatTensor(std)
    if torch.cuda.is_available():
        mean, std = mean.cuda(), std.cuda()
    mean, std = mean.view(1,-1,1,1), std.view(1,-1,1,1)

    return (images - mean) / std


def get_loader(args):
    if NSML_NFS_OUTPUT:
        root = os.path.join(NSML_NFS_OUTPUT, args.data_dir)
    elif DATASET_PATH:
        assert HAS_DATASET, "Can't find dataset in nsml. Push or search the dataset"
        root = os.path.join(DATASET_PATH, 'train')
    else:
        root = os.path.join(PROJECT_ROOT.as_posix(), args.data_dir)

    train_loader, val_loader = (torch.utils.data.DataLoader(
        globals()[args.dataset](root=root, train=is_training).preprocess(),
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
        ) for is_training in [True, False]
    )
    return train_loader, val_loader


class MNIST(datasets.MNIST):
    def __init__(self, root, train, download=False):
        super().__init__(root, train=train, download=download)

    def preprocess(self):
        mean, std = data_stats("MNIST")
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        return self

    def __getitem__(self, idx):
        image, label = super(MNIST, self).__getitem__(idx)
        return image, one_hot(label, 10)


class CIFAR10(datasets.CIFAR10):
    def __init__(self, root, train, download=False):
        super().__init__(root, train=train, download=download)

    def preprocess(self):
        mean, std = data_stats("CIFAR10")
        normalize = transforms.Normalize(mean=mean, std=std)
        if self.train:
            self.transform = transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32, 4),
                                transforms.ToTensor(),
                                normalize,
                            ])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(), normalize,])
        return self

    def __getitem__(self, idx):
        image, label = super(CIFAR10, self).__getitem__(idx)
        return image, one_hot(label, 10)


class CIFAR100(datasets.CIFAR100):
    def __init__(self, root, train, download=False):
        super().__init__(root, train=train, download=download)

    def preprocess(self):
        mean, std = data_stats("CIFAR10")
        normalize = transforms.Normalize(mean=mean, std=std)
        if self.train:
            self.transform = transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32, 4),
                                transforms.ToTensor(),
                                normalize,
                            ])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(), normalize,])
        return self

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        return image, one_hot(label, 100)


class ImageNet(datasets.ImageFolder):
    def __init__(self, root, train, download=False):
        root = os.path.join(root, "Data", "CLS-LOC")
        self.train = train
        if self.train: root = os.path.join(root, "train")
        else: root = os.path.join(root, "val")
        super(ImageNet, self).__init__(root=root)

    def preprocess(self):
        mean, std = data_stats("ImageNet")
        normalize = transforms.Normalize(mean=mean, std=std)
        if self.train:
            self.transform = transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize,
                            ])
        else:
            self.transform = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize,
                            ])
        return self

    def __getitem__(self, idx):
        image, label = super(ImageNet, self).__getitem__(idx)
        return image, one_hot(label, 1000)


class TinyImageNet(datasets.ImageFolder):
    def __init__(self, root, train, download=False):
        self.train = train
        if self.train:
            root = os.path.join(root, "train")
        else:
            root = os.path.join(root, "val")
            ann = os.path.join(root, "val_annotations.txt")
            if os.path.isfile(ann):
                self.create_dir(root, ann)
        super(TinyImageNet, self).__init__(root=root)

    def create_dir(self, root, ann):
        img_path = os.path.join(root, "images")
        img_labels = {}
        with open(ann, "r") as f:
            data = f.readlines()
            for line in data:
                words = line.split('\t')
                img_labels[words[0]] = words[1]

        for img, label in img_labels.items():
            new_path = os.path.join(root, label)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            if os.path.exists(os.path.join(img_path, img)):
                os.rename(os.path.join(img_path, img), os.path.join(new_path, img))
        os.rmdir(os.path.join(root, "images"))
        os.remove(ann)

    def preprocess(self):
        mean, std = data_stats("ImageNet")
        normalize = transforms.Normalize(mean=mean, std=std)
        if self.train:
            self.transform = transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize,
                            ])
        else:
            self.transform = transforms.Compose([
                                transforms.ToTensor(),
                                normalize,
                            ])
        return self

    def __getitem__(self, idx):
        image, label = super(TinyImageNet, self).__getitem__(idx)
        return image, one_hot(label, 200)


class COCO(torch.utils.data.Dataset):
    # NOTE: used only for autoencoder training
    def __init__(self, root, train, download=False):
        root = os.path.join(root, 'images', 'trainval35k')
        self.train = train
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        label_fake = torch.zeros(1)
        return image, label_fake
    
    def __len__(self):
        return len(self.image_paths)

    def preprocess(self):
        mean, std = data_stats("ImageNet")
        normalize = transforms.Normalize(mean=mean, std=std)
        if self.train:
            self.transform = transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize,
                            ])
        else:
            self.transform = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize,
                            ])
        return self
