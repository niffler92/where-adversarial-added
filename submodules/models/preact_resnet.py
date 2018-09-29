'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from submodules.activation import get_activation, CustomNorm2d

__all__ = ['PreActResNet18', 'PreActResNet34', 'PreActResNet50', 'PreActResNet101',
           'PreActResNet152']


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, nlayer=0, nstride=0, args=None, **kwargs):
        super(PreActBlock, self).__init__()
        self.bn1 = CustomNorm2d(in_planes, args, **kwargs)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = CustomNorm2d(planes, args, **kwargs)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

        name = 'Block_' + str(nlayer) + '_' + str(nstride) + '_'
        self.activation1 = get_activation(args.activation, name+'bn1', args, **kwargs)
        self.activation2 = get_activation(args.activation, name+'conv2', args, **kwargs)

    def forward(self, x):
        out = self.activation1(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.activation2(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, nlayer=0, nstride=0, args=None, **kwargs):
        super(PreActBottleneck, self).__init__()
        self.bn1 = CustomNorm2d(in_planes, args, **kwargs)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = CustomNorm2d(planes, args, **kwargs)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = CustomNorm2d(planes, args, **kwargs)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

        name = 'Bottleneck_' + str(nlayer) + '_' + str(nstride) + '_'
        self.activation1 = get_activation(args.activation, name+'bn1', args, **kwargs)
        self.activation2 = get_activation(args.activation, name+'conv2', args, **kwargs)
        self.activation3 = get_activation(args.activation, name+'conv3', args, **kwargs)

    def forward(self, x):
        out = self.activation1(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.activation2(self.bn2(out)))
        out = self.conv3(self.activation3(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, args, **kwargs):
        super(PreActResNet, self).__init__()
        self.args = args
        self.kwargs = kwargs

        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, nlayer=0)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, nlayer=1)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, nlayer=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, nlayer=3)
        self.linear = nn.Linear(512*block.expansion, args.num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, nlayer):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride, nlayer, i, self.args, **self.kwargs))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreActResNet18(args, **kwargs):
    return PreActResNet(PreActBlock, [2,2,2,2], args, **kwargs)

def PreActResNet34(args, **kwargs):
    return PreActResNet(PreActBlock, [3,4,6,3], args, **kwargs)

def PreActResNet50(args, **kwargs):
    return PreActResNet(PreActBottleneck, [3,4,6,3], args, **kwargs)

def PreActResNet101(args, **kwargs):
    return PreActResNet(PreActBottleneck, [3,4,23,3], args, **kwargs)

def PreActResNet152(args, **kwargs):
    return PreActResNet(PreActBottleneck, [3,8,36,3], args, **kwargs)


def test():
    net = PreActResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test()
