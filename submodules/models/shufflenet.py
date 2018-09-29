'''ShuffleNet in PyTorch.

See the paper "ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from submodules.activation import get_activation, CustomNorm2d

__all__ = ['ShuffleNetG2', 'ShuffleNetG3']


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        return x.view(N,g,int(C/g),H,W).permute(0,2,1,3,4).contiguous().view(N,C,H,W)


class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups, nlayer, nblock, args, **kwargs):
        super(Bottleneck, self).__init__()
        self.stride = stride

        mid_planes = int(out_planes/4)
        g = 1 if in_planes == 24 else groups
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)
        self.bn1 = CustomNorm2d(mid_planes, args, **kwargs)
        self.shuffle1 = ShuffleBlock(groups=g)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)
        self.bn2 = CustomNorm2d(mid_planes, args, **kwargs)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
        self.bn3 = CustomNorm2d(out_planes, args, **kwargs)

        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

        name = 'Bottleneck_' + str(nlayer) + '_' + str(nblock) + '_'
        self.activation1 = get_activation(args.activation, name+'conv1', args, **kwargs)
        self.activation2 = get_activation(args.activation, name+'conv2', args, **kwargs)
        self.activation3 = get_activation(args.activation, name+'out', args, **kwargs)

    def forward(self, x):
        out = self.activation1(self.bn1(self.conv1(x)))
        out = self.shuffle1(out)
        out = self.activation2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        res = self.shortcut(x)
        out = self.activation3(torch.cat([out,res], 1)) if self.stride==2 else self.activation3(out+res)
        return out


class ShuffleNet(nn.Module):
    def __init__(self, cfg, args, **kwargs):
        super(ShuffleNet, self).__init__()
        self.args = args
        self.kwargs = kwargs
        self.activation = get_activation(args.activation, 'conv1', args, **kwargs)

        out_planes = cfg['out_planes']
        num_blocks = cfg['num_blocks']
        groups = cfg['groups']

        self.conv1 = nn.Conv2d(3, 24, kernel_size=1, bias=False)
        self.bn1 = CustomNorm2d(24, args, **kwargs)
        self.in_planes = 24
        self.layer1 = self._make_layer(out_planes[0], num_blocks[0], groups, nlayer=0)
        self.layer2 = self._make_layer(out_planes[1], num_blocks[1], groups, nlayer=1)
        self.layer3 = self._make_layer(out_planes[2], num_blocks[2], groups, nlayer=2)
        self.linear = nn.Linear(out_planes[2], args.num_classes)

    def _make_layer(self, out_planes, num_blocks, groups, nlayer):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            cat_planes = self.in_planes if i == 0 else 0
            layers.append(
                Bottleneck(self.in_planes,
                           out_planes-cat_planes,
                           stride=stride,
                           groups=groups,
                           nlayer=nlayer,
                           nblock=i,
                           args=self.args,
                           **self.kwargs)
            )
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ShuffleNetG2(args, **kwargs):
    cfg = {
        'out_planes': [200,400,800],
        'num_blocks': [4,8,4],
        'groups': 2
    }
    return ShuffleNet(cfg, args, **kwargs)

def ShuffleNetG3(args, **kwargs):
    cfg = {
        'out_planes': [240,480,960],
        'num_blocks': [4,8,4],
        'groups': 3
    }
    return ShuffleNet(cfg, args, **kwargs)


def test():
    net = ShuffleNetG2()
    x = Variable(torch.randn(1,3,32,32))
    y = net(x)
    print(y)

# test()
