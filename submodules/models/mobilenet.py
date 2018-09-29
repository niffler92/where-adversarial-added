'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from submodules.activation import get_activation, CustomNorm2d

__all__ = ['MobileNet']


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1, nlayer=None, args=None, **kwargs):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = CustomNorm2d(in_planes, args, **kwargs)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = CustomNorm2d(out_planes, args, **kwargs)

        name = 'Block_' + str(nlayer) + '_'
        self.activation1 = get_activation(args.activation, name+'conv1', args, **kwargs)
        self.activation2 = get_activation(args.activation, name+'conv2', args, **kwargs)

    def forward(self, x):
        out = self.activation1(self.bn1(self.conv1(x)))
        out = self.activation2(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, args=None, **kwargs):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = CustomNorm2d(32, args, **kwargs)
        self.layers = self._make_layers(in_planes=32, args=args, **kwargs)
        self.linear = nn.Linear(1024, args.num_classes)

        self.activation = get_activation(args.activation, 'conv1', args, **kwargs)

    def _make_layers(self, in_planes, args, **kwargs):
        layers = []
        for i, x in enumerate(self.cfg):
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride, i, args, **kwargs))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = MobileNet()
    x = torch.randn(1,3,32,32)
    y = net(Variable(x))
    print(y.size())

# test()