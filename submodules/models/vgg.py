'''
Modified from https://github.com/pytorch/vision.git
'''
import math
import sys

import torch.nn as nn
from torch.nn import Sequential
import torch.nn.init as init


__all__ = ['Vgg11', 'Vgg13', 'Vgg16', 'Vgg19']


class VGG(nn.Module):
    '''
    VGG model - For monotonic activation function.
    '''
    def __init__(self, cfg, args=None, **kwargs):
        super(VGG, self).__init__()
        if args.dataset in ["CIFAR10", "CIFAR100"]:
            linear_in = 512
        elif args.dataset == "TinyImageNet":
            linear_in = 512*2*2
        elif args.dataset == "ImageNet":
            linear_in = 512*7*7
        else:
            raise NotImplementedError

        self.activation = args.activation
        self.args = args
        self.kwargs = kwargs
        self.features = self.make_layers(cfg)

        self.classifier = Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(linear_in, 512),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, args.num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def make_layers(self, cfg):
        layers = []
        in_channels = 3
        for i, v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(v),
                           nn.ReLU()]
                in_channels = v
        return Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def Vgg11(args, **kwargs):
    """VGG 11-layer model (configuration "A")"""
    return VGG(cfg['A'], args=args, **kwargs)

def Vgg13(args, **kwargs):
    """VGG 13-layer model (configuration "B")"""
    return VGG(cfg['B'], args=args, **kwargs)

def Vgg16(args, **kwargs):
    """VGG 16-layer model (configuration "D")"""
    return VGG(cfg['D'], args=args, **kwargs)

def Vgg19(args, **kwargs):
    """VGG 19-layer model (configuration "E")"""
    return VGG(cfg['E'], args=args, **kwargs)
