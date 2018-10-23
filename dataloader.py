import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable


def data_stats(dataset):
    """
    Returns: *mean, *std
    """
    if dataset in ["CIFAR10", "CIFAR100", "TinyImageNet", "ImageNet"]:
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
    from common.torch_utils import to_var

    mean, std = data_stats(dataset)
    mean, std = torch.FloatTensor(mean), torch.FloatTensor(std)
    if torch.cuda.is_available():
        mean, std = mean.cuda(), std.cuda()
    mean, std = mean.view(1,-1,1,1), std.view(1,-1,1,1)

    if isinstance(images, torch.autograd.Variable):
        mean, std = Variable(mean), Variable(std)

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
    from common.torch_utils import to_var

    mean, std = data_stats(dataset)
    mean, std = torch.FloatTensor(mean), torch.FloatTensor(std)
    if torch.cuda.is_available():
        mean, std = mean.cuda(), std.cuda()
    mean, std = mean.view(1,-1,1,1), std.view(1,-1,1,1)

    if isinstance(images, torch.autograd.Variable):
        mean, std = Variable(mean), Variable(std)

    return (images - mean) / std


def get_loader(
        dataset='CIFAR10',
        root='./data',
        batch_size=128,
        num_workers=4
        ):

    assert dataset in ['CIFAR10', 'CIFAR100', 'MNIST', 'ImageNet', 'TinyImageNet'], "Unknown dataset"

    train_loader, val_loader = (torch.utils.data.DataLoader(
        globals()[dataset](root=root, train=is_training).preprocess(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
        ) for is_training in [True, False]
    )
    return train_loader, val_loader


class MNIST(datasets.MNIST):
    def __init__(self, root, train, download=True):
        super().__init__(root, train=train, download=download)

    def preprocess(self):
        mean, std = data_stats("MNIST")
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        return self


class CIFAR10(datasets.CIFAR10):
    def __init__(self, root, train, download=True):
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


class CIFAR100(CIFAR10):
    base_folder = 'cifar-100-python'
    url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]


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


if __name__ == '__main__':
    eps = 0.031
    x = torch.zeros([1, 3, 32, 32]) + eps
    print(x)
    print(normalize(x, "CIFAR10"))
    print(denormalize(normalize(x, "CIFAR10"), "CIFAR10"))
