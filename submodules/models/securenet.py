import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential
from torch.autograd import Variable

from submodules.activation import get_activation, CustomNorm2d
from submodules.splitconv import SplitConv2d
from submodules.models.resnet import BasicBlock, Bottleneck, ResNet


__all__ = ['SecureNet18']


class SecureBlock(BasicBlock):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, nlayer=0, nstride=0, args=None, **kwargs):
        super(SecureBlock, self).__init__(in_planes, planes, stride=stride,
                                          nlayer=nlayer, nstride=nstride, args=args, **kwargs)
        if '2x2' in args.model and stride == 2:
            ks = 2
            padding = 0
        elif '4x4' in args.model and stride == 2:
            ks = 4
            padding = 1
        else:
            ks = 3
            padding = 1

        self.conv1 = SplitConv2d(in_planes, planes, kernel_size=ks, stride=stride, padding=padding)
        self.bn1 = CustomNorm2d(planes, args, **kwargs)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = CustomNorm2d(planes, args, **kwargs)

        self.shortcut = Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                CustomNorm2d(self.expansion*planes, args, **kwargs)
            )

        name = 'SecureBlock_' + str(nlayer) + '_' + str(nstride) + '_'
        self.activation1 = get_activation(args.activation, name+'conv1', args, **kwargs)
        self.activation2 = get_activation(args.activation, name+'out', args, **kwargs)


class Secureneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, nlayer=0, nstride=0, args=None, **kwargs):
        super(Secureneck, self).__init__(in_planes, planes, stride=stride,
                                          nlayer=nlayer, nstride=nstride, args=args, **kwargs)
        if '2x2' in args.model and stride == 2:
            ks = 2
            padding = 0
        elif '4x4' in args.model and stride == 2:
            ks = 4
            padding = 1
        else:
            ks = 3
            padding = 1

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = CustomNorm2d(planes, args, **kwargs)
        self.conv2 = SplitConv2d(planes, planes, kernel_size=ks, stride=stride, padding=padding)
        self.bn2 = CustomNorm2d(planes, args, **kwargs)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = CustomNorm2d(self.expansion*planes, args, **kwargs)

        self.shortcut = Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                CustomNorm2d(self.expansion*planes, args, **kwargs)
            )

        name = 'Bottleneck_' + str(nlayer) + '_' + str(nstride) + '_'
        self.activation1 = get_activation(args.activation, name+'conv1', args, **kwargs)
        self.activation2 = get_activation(args.activation, name+'conv2', args, **kwargs)
        self.activation3 = get_activation(args.activation, name+'out', args, **kwargs)


class SecureNet(ResNet):
    def __init__(self, block, num_blocks, args=None, **kwargs):
        super(SecureNet, self).__init__(block, num_blocks, args=args, **kwargs)
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = CustomNorm2d(64, args, **kwargs)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, nlayer=0, args=args, **kwargs)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, nlayer=1, args=args, **kwargs)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, nlayer=2, args=args, **kwargs)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, nlayer=3, args=args, **kwargs)
        self.linear = nn.Linear(512*block.expansion, args.num_classes)
        self.activation = get_activation(args.activation, 'conv1', args, **kwargs)


def SecureNet18(args, **kwargs):
    return SecureNet(SecureBlock, [2,2,2,2], args, **kwargs)
