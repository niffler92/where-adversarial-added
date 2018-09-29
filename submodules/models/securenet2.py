import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential
from torch.autograd import Variable

from submodules.activation import get_activation, CustomNorm2d
from submodules.splitconv2 import SplitConv2d_3, SplitConv2d


__all__ = ['SecureNet18_2x2_3', 'SecureNet18_2x2']

class BasicBlock3(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, nlayer=0, nstride=0, args=None, **kwargs):
        super(BasicBlock3, self).__init__()
        self.stride = stride
        if '2x2' in args.model and stride == 2:
            ks = 2
            padding = 0
        elif '4x4' in args.model and stride == 2:
            ks = 4
            padding = 1
        else:
            ks = 3
            padding = 1

        self.conv1 = SplitConv2d_3(in_planes, planes, kernel_size=ks, stride=stride, padding=padding, bias=False)
        #self.bn1 = CustomNorm2d(planes, args, **kwargs)
        if stride != 2:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = CustomNorm2d(planes, args, **kwargs)

        self.shortcut = Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=2, stride=stride, bias=False),
                CustomNorm2d(self.expansion*planes, args, **kwargs)
            )

        name = 'Block_' + str(nlayer) + '_' + str(nstride) + '_'
        self.activation1 = get_activation(args.activation, name+'conv1', args, **kwargs)
        self.activation2 = get_activation(args.activation, name+'out', args, **kwargs)

    def forward(self, x):
        out = self.conv1(x)
        if self.stride != 2:
            out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation2(out)
        return out


class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, nlayer=0, nstride=0, args=None, **kwargs):
        super(BasicBlock2, self).__init__()
        if '2x2' in args.model and stride == 2:
            ks = 2
            padding = 0
        elif '4x4' in args.model and stride == 2:
            ks = 4
            padding = 1
        else:
            ks = 3
            padding = 1

        self.conv1 = SplitConv2d(in_planes, planes, kernel_size=ks, stride=stride, padding=padding, bias=False)
        #self.bn1 = CustomNorm2d(planes, args, **kwargs)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = CustomNorm2d(planes, args, **kwargs)

        self.shortcut = Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=2, stride=stride, bias=False),
                CustomNorm2d(self.expansion*planes, args, **kwargs)
            )

        name = 'Block_' + str(nlayer) + '_' + str(nstride) + '_'
        self.activation1 = get_activation(args.activation, name+'conv1', args, **kwargs)
        self.activation2 = get_activation(args.activation, name+'out', args, **kwargs)

    def forward(self, x):
        out = self.activation1(self.conv1(x))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, nlayer=0, nstride=0, args=None, **kwargs):
        super(Bottleneck, self).__init__()
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
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=ks, stride=stride, padding=padding, bias=False)
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

    def forward(self, x):
        out = self.activation1(self.bn1(self.conv1(x)))
        out = self.activation2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.activation3(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, args=None, **kwargs):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = CustomNorm2d(64, args, **kwargs)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, nlayer=0, args=args, **kwargs)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, nlayer=1, args=args, **kwargs)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, nlayer=2, args=args, **kwargs)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, nlayer=3, args=args, **kwargs)
        self.linear = nn.Linear(512*block.expansion, args.num_classes)

        self.activation = get_activation(args.activation, 'conv1', args, **kwargs)

    def _make_layer(self, block, planes, num_blocks, stride, nlayer, args, **kwargs):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride, nlayer, i, args, **kwargs))
            self.in_planes = planes * block.expansion
        return Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = F.avg_pool2d(out4, 4)
        out5 = out.view(out5.size(0), -1)
        out6 = self.linear(out5)

        return out6


class ResNet_g(nn.Module):
    def __init__(self, block, num_blocks, args=None, **kwargs):
        super(ResNet_g, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = CustomNorm2d(64, args, **kwargs)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, nlayer=0, args=args, **kwargs)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, nlayer=1, args=args, **kwargs)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, nlayer=2, args=args, **kwargs)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, nlayer=3, args=args, **kwargs)
        self.linear = nn.Linear(512*block.expansion, args.num_classes)

        self.activation = get_activation(args.activation, 'conv1', args, **kwargs)

        self.grads = {}
        self.get_grad = False
        self.endpoints = {}

    def set_grad(self):
        self.get_grad = True

    def save_grad(self, name):
        def hook(grad):
            self.grads[name] = grad
        return hook

    def _make_layer(self, block, planes, num_blocks, stride, nlayer, args, **kwargs):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride, nlayer, i, args, **kwargs))
            self.in_planes = planes * block.expansion
        return Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = F.avg_pool2d(out4, 4)
        out5 = out5.view(out5.size(0), -1)
        out6 = self.linear(out5)

        if x.requires_grad:
            x.register_hook(self.save_grad('input'))
            out.register_hook(self.save_grad('conv1'))
            out1.register_hook(self.save_grad('block1'))
            out2.register_hook(self.save_grad('block2'))
            out3.register_hook(self.save_grad('block3'))
            out4.register_hook(self.save_grad('block4'))

            self.endpoints['input'] = x
            self.endpoints['conv1'] = out
            self.endpoints['block1'] = out1
            self.endpoints['block2'] = out2
            self.endpoints['block3'] = out3
            self.endpoints['block4'] = out4

        return out6


def SecureNet18_2x2(args, **kwargs):
    return ResNet_g(BasicBlock2, [2,2,2,2], args, **kwargs)

def SecureNet18_2x2_3(args, **kwargs):
    return ResNet_g(BasicBlock3, [2,2,2,2], args, **kwargs)
