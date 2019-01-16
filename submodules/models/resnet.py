'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential

__all__ = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, nlayer=0, nstride=0, args=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        name = 'Block_' + str(nlayer) + '_' + str(nstride) + '_'
        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()

    def forward(self, x):
        out = self.activation1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, nlayer=0, nstride=0, args=None, **kwargs):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        name = 'Bottleneck_' + str(nlayer) + '_' + str(nstride) + '_'
        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()
        self.activation3 = nn.ReLU()

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
        if args.dataset == "MNIST":
            conv_in = 1
            linear_in = 512
        else:
            conv_in = 3
            if args.dataset in ["CIFAR10", "CIFAR100"]:
                linear_in = 512
            elif args.dataset == "TinyImageNet":
                linear_in = 512*2*2
            elif args.dataset == "ImageNet":
                linear_in = 512*7*7
            else:
                raise NotImplementedError

        self.conv1 = nn.Conv2d(conv_in, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, nlayer=0, args=args, **kwargs)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, nlayer=1, args=args, **kwargs)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, nlayer=2, args=args, **kwargs)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, nlayer=3, args=args, **kwargs)
        self.linear = nn.Linear(linear_in*block.expansion, args.num_classes)

        self.activation = nn.ReLU()

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

        return out6


def ResNet18(args, **kwargs):
    return ResNet(BasicBlock, [2,2,2,2], args, **kwargs)

def ResNet34(args, **kwargs):
    return ResNet(BasicBlock, [3,4,6,3], args, **kwargs)

def ResNet50(args, **kwargs):
    return ResNet(Bottleneck, [3,4,6,3], args, **kwargs)

def ResNet101(args, **kwargs):
    return ResNet(Bottleneck, [3,4,23,3], args, **kwargs)

def ResNet152(args, **kwargs):
    return ResNet(Bottleneck, [3,8,36,3], args, **kwargs)
