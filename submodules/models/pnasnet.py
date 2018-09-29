'''PNASNet in PyTorch.

Paper: Progressive Neural Architecture Search
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from submodules.activation import get_activation, CustomNorm2d

__all__ = ['PNASNetA', 'PNASNetB']


class SepConv(nn.Module):
    '''Separable Convolution.'''
    def __init__(self, in_planes, out_planes, kernel_size, stride, args, **kwargs):
        super(SepConv, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes,
                               kernel_size, stride,
                               padding=(kernel_size-1)//2,
                               bias=False, groups=in_planes)
        self.bn1 = CustomNorm2d(out_planes, args, **kwargs)

    def forward(self, x):
        return self.bn1(self.conv1(x))


class CellA(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nlayer=0, ncell=0, args=None, **kwargs):
        super(CellA, self).__init__()
        self.stride = stride
        self.sep_conv1 = SepConv(in_planes, out_planes, kernel_size=7, stride=stride, args=args, **kwargs)
        if stride==2:
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn1 = CustomNorm2d(out_planes, args, **kwargs)

        self.activation = get_activation(args.activation, 'CellA_'+str(nlayer)+'_'+str(ncell)+'_out', args, **kwargs)

    def forward(self, x):
        y1 = self.sep_conv1(x)
        y2 = F.max_pool2d(x, kernel_size=3, stride=self.stride, padding=1)
        if self.stride==2:
            y2 = self.bn1(self.conv1(y2))
        return self.activation(y1+y2)

class CellB(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nlayer=0, ncell=0, args=None, **kwargs):
        super(CellB, self).__init__()
        self.stride = stride
        # Left branch
        self.sep_conv1 = SepConv(in_planes, out_planes, kernel_size=7, stride=stride, args=args, **kwargs)
        self.sep_conv2 = SepConv(in_planes, out_planes, kernel_size=3, stride=stride, args=args, **kwargs)
        # Right branch
        self.sep_conv3 = SepConv(in_planes, out_planes, kernel_size=5, stride=stride, args=args, **kwargs)
        if stride==2:
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn1 = CustomNorm2d(out_planes, args, **kwargs)
        # Reduce channels
        self.conv2 = nn.Conv2d(2*out_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = CustomNorm2d(out_planes, args, **kwargs)

        self.activation1 = get_activation(args.activation, 'CellB_'+str(nlayer)+'_'+str(ncell)+'_conv1', args, **kwargs)
        self.activation2 = get_activation(args.activation, 'CellB_'+str(nlayer)+'_'+str(ncell)+'_conv2', args, **kwargs)
        self.activation3 = get_activation(args.activation, 'CellB_'+str(nlayer)+'_'+str(ncell)+'_out', args, **kwargs)

    def forward(self, x):
        # Left branch
        y1 = self.sep_conv1(x)
        y2 = self.sep_conv2(x)
        # Right branch
        y3 = F.max_pool2d(x, kernel_size=3, stride=self.stride, padding=1)
        if self.stride==2:
            y3 = self.bn1(self.conv1(y3))
        y4 = self.sep_conv3(x)
        # Concat & reduce channels
        b1 = self.activation1(y1+y2)
        b2 = self.activation2(y3+y4)
        y = torch.cat([b1,b2], 1)
        return self.activation3(self.bn2(self.conv2(y)))

class PNASNet(nn.Module):
    def __init__(self, cell_type, num_cells, num_planes, args, **kwargs):
        super(PNASNet, self).__init__()
        self.args = args
        self.kwargs = kwargs
        self.activation = get_activation(args.activation, 'conv1', args, **kwargs)

        self.in_planes = num_planes
        self.cell_type = cell_type

        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = CustomNorm2d(num_planes, args, **kwargs)

        self.layer1 = self._make_layer(num_planes, num_cells=6, nlayer=0)
        self.layer2 = self._downsample(num_planes*2, nlayer=1)
        self.layer3 = self._make_layer(num_planes*2, num_cells=6, nlayer=2)
        self.layer4 = self._downsample(num_planes*4, nlayer=3)
        self.layer5 = self._make_layer(num_planes*4, num_cells=6, nlayer=4)

        self.linear = nn.Linear(num_planes*4, args.num_classes)

    def _make_layer(self, planes, num_cells, nlayer):
        layers = []
        for i, _ in enumerate(range(num_cells)):
            layers.append(self.cell_type(self.in_planes, planes, stride=1, nlayer=nlayer, ncell=i, args=self.args, **self.kwargs))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _downsample(self, planes, nlayer):
        layer = self.cell_type(self.in_planes, planes, stride=2, nlayer=nlayer, args=self.args, **self.kwargs)
        self.in_planes = planes
        return layer

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = F.avg_pool2d(out, 8)
        out = self.linear(out.view(out.size(0), -1))
        return out


def PNASNetA(args, **kwargs):
    return PNASNet(CellA, num_cells=6, num_planes=44, args=args, **kwargs)

def PNASNetB(args, **kwargs):
    return PNASNet(CellB, num_cells=6, num_planes=32, args=args, **kwargs)


def test():
    net = PNASNetB()
    print(net)
    x = Variable(torch.randn(1,3,32,32))
    y = net(x)
    print(y)

# test()
