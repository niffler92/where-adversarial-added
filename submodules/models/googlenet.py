'''GoogLeNet with PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from common.torch_utils import get_activation

__all__ = ['GoogLeNet']


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes, name, args=None, **kwargs):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            get_activation(args.activation, args, **kwargs)
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            get_activation(args.activation, args, **kwargs)
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            n3x3 args, **kwargs),
            get_activation(args.activation, args, **kwargs)
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            get_activation(args.activation, args, **kwargs)
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            get_activation(args.activation, args, **kwargs)
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            get_activation(args.activation, args, **kwargs)
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            get_activation(args.activation, args, **kwargs)
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self, args=None, **kwargs):
        super(GoogLeNet, self).__init__()
        self.activation = get_activation(args.activation, args, **kwargs)

        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            self.activation,
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32, 'a3', args, **kwargs)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64, 'b3', args, **kwargs)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64, 'a4', args, **kwargs)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64, 'b4', args, **kwargs)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64, 'c4', args, **kwargs)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64, 'd4', args, **kwargs)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128, 'e4', args, **kwargs)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128, 'a5', args, **kwargs)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128, 'b5', args, **kwargs)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, args.num_classes)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# net = GoogLeNet()
# x = torch.randn(1,3,32,32)
# y = net(Variable(x))
# print(y.size())
