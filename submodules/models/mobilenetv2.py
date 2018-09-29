'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from submodules.activation import get_activation, CustomNorm2d

__all__ = ['MobileNetV2']


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride, nblock, nstride, args, **kwargs):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = CustomNorm2d(planes, args, **kwargs)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = CustomNorm2d(planes, args, **kwargs)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = CustomNorm2d(out_planes, args, **kwargs)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                CustomNorm2d(out_planes, args, **kwargs),
            )

        name = 'Block_' + str(nblock) + '_' + str(nstride) + '_'
        self.activation1 = get_activation(args.activation, name+'conv1', args, **kwargs)
        self.activation2 = get_activation(args.activation, name+'conv2', args, **kwargs)

    def forward(self, x):
        out = self.activation1(self.bn1(self.conv1(x)))
        out = self.activation2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, args, **kwargs):
        super(MobileNetV2, self).__init__()
        self.args = args
        self.activation1 = get_activation(args.activation, 'conv1', args, **kwargs)
        self.activation2 = get_activation(args.activation, 'conv2', args, **kwargs)
        self.kwargs = kwargs

        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = CustomNorm2d(32, args, **kwargs)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = CustomNorm2d(1280, args, **kwargs)
        self.linear = nn.Linear(1280, args.num_classes)


    def _make_layers(self, in_planes):
        layers = []
        for i, (expansion, out_planes, num_blocks, stride) in enumerate(self.cfg):
            strides = [stride] + [1]*(num_blocks-1)
            for j, stride in enumerate(strides):
                layers.append(Block(in_planes, out_planes, expansion, stride, i, j, self.args, **self.kwargs))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation1(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.activation2(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = MobileNetV2()
    x = Variable(torch.randn(2,3,32,32))
    y = net(x)
    print(y.size())

# test()
