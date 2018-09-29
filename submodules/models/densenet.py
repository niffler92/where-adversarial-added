'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from submodules.activation import get_activation, CustomNorm2d

__all__ = ['DenseNet121', 'DenseNet161', 'DenseNet169', 'DenseNet201', 'densenet_cifar']


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate, ndense, nblock, args=None, **kwargs):
        super(Bottleneck, self).__init__()
        self.bn1 = CustomNorm2d(in_planes, args, **kwargs)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = CustomNorm2d(4*growth_rate, args, **kwargs)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

        name = 'Bottleneck_'+str(ndense)+'_'+str(nblock)+'_'
        self.activation1 = get_activation(args.activation, name + 'conv1', args, **kwargs)
        self.activation2 = get_activation(args.activation, name + 'conv2', args, **kwargs)

    def forward(self, x):
        out = self.conv1(self.activation1(self.bn1(x)))
        out = self.conv2(self.activation2(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes, ntrans, args=None, **kwargs):
        super(Transition, self).__init__()
        self.bn = CustomNorm2d(in_planes, args, **kwargs)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

        name = 'Transition_'+str(ntrans)+'_'
        self.activation = get_activation(args.activation, name+'conv1', args, **kwargs)

    def forward(self, x):
        out = self.conv(self.activation(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, args=None, **kwargs):
        super(DenseNet, self).__init__()
        self.args = args
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0], 0, args, **kwargs)
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes, 0, args, **kwargs)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1], 1, args, **kwargs)
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes, 1, args, **kwargs)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2], 2, args, **kwargs)
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes, 2, args, **kwargs)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3], 3, args, **kwargs)
        num_planes += nblocks[3]*growth_rate

        self.bn = CustomNorm2d(num_planes, args, **kwargs)
        self.linear = nn.Linear(num_planes, self.args.num_classes)

        self.activation = get_activation(args.activation, 'out', args, **kwargs)

    def _make_dense_layers(self, block, in_planes, nblock, ndense, args, **kwargs):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate, ndense, i, args, **kwargs))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(self.activation(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def DenseNet121(args, **kwargs):
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32, args=args, **kwargs)

def DenseNet169(args, **kwargs):
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32, args=args, **kwargs)

def DenseNet201(args, **kwargs):
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32, args=args, **kwargs)

def DenseNet161(args, **kwargs):
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48, args=args, **kwargs)

def densenet_cifar(args, **kwargs):
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12, args=args, **kwargs)

def test_densenet():
    net = densenet_cifar()
    x = torch.randn(1,3,32,32)
    y = net(Variable(x))
    print(y)

# test_densenet()
