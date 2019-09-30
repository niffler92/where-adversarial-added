'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['LeNet_toy']


class LeNet_toy(nn.Module):
    def __init__(self, args, **kwargs):
        super(LeNet_toy, self).__init__()
        if args.dataset == "MNIST":
            conv_in = 1
            linear_in = 1024
        else:
            conv_in = 3
            if args.dataset in ["CIFAR10", "CIFAR100"]:
                linear_in = 1600
            elif args.dataset == "TinyImageNet":
                linear_in = 1600*2*2
            elif args.dataset == "ImageNet":
                linear_in = 1600*7*7
            else:
                raise NotImplementedError

        self.conv1 = nn.Conv2d(conv_in, 16, 3, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1)

        self.fc1   = nn.Linear(linear_in, 512)
        self.fc2   = nn.Linear(512, args.num_classes)

        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()
        self.activation3 = nn.ReLU()
        self.activation4 = nn.ReLU()

        self.grads = {}
        self.get_grad = False
        self.endpoints = {}

    def set_grad(self):
        self.get_grad = True

    def save_grad(self, name):
        def hook(grad):
            self.grads[name] = grad
        return hook

    def forward(self, x):
        conv1 = self.activation1(self.conv1(x))
        conv2 = self.activation2(self.conv2(conv1))
        conv3 = self.activation3(self.conv3(conv2))
        out = conv3.view(conv3.size(0), -1)
        fc1 = self.activation4(self.fc1(out))
        out = self.fc2(fc1)

        if x.requires_grad:
            x.register_hook(self.save_grad('input'))
            conv1.register_hook(self.save_grad('conv1'))
            conv2.register_hook(self.save_grad('conv2'))
            conv3.register_hook(self.save_grad('conv3'))

            self.endpoints['input'] = x
            self.endpoints['conv1'] = conv1
            self.endpoints['conv2'] = conv2
            self.endpoints['conv3'] = conv3

        return out
