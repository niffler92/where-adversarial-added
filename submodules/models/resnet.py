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
from torch.autograd import Variable

from common.torch_utils import get_activation


__all__ = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']
__all__ += ['ResNet18_g', 'ResNet34_g', 'ResNet50_g', 'ResNet101_g', 'ResNet152_g']
__all__ += ['ResNet18_2x2', 'ResNet18_4x4', 'ResNet101_2x2', 'ResNet101_4x4']
__all__ += ['ResNet18_hook']


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, nlayer=0, nstride=0, args=None, **kwargs):
        super(BasicBlock, self).__init__()
        if '2x2' in args.model and stride == 2:
            ks = 2
            padding = 0
        elif '4x4' in args.model and stride == 2:
            ks = 4
            padding = 1
        else:
            ks = 3
            padding = 1

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            shortcut_ks = 2 if '2x2' in args.model or '4x4' in args.model else 1
            self.shortcut = Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=shortcut_ks, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        name = 'Block_' + str(nlayer) + '_' + str(nstride) + '_'
        self.activation1 = get_activation(args.activation, args, **kwargs)
        self.activation2 = get_activation(args.activation, args, **kwargs)

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
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=ks, stride=stride, padding=padding, bias=False)
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
        self.activation1 = get_activation(args.activation, args, **kwargs)
        self.activation2 = get_activation(args.activation, args, **kwargs)
        self.activation3 = get_activation(args.activation, args, **kwargs)

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
        if args.dataset in ["CIFAR10", "CIFAR100"]:
            linear_in = 512
        elif args.dataset == "TinyImageNet":
            linear_in = 512*2*2
        elif args.dataset == "ImageNet":
            linear_in = 512*7*7
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, nlayer=0, args=args, **kwargs)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, nlayer=1, args=args, **kwargs)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, nlayer=2, args=args, **kwargs)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, nlayer=3, args=args, **kwargs)
        self.linear = nn.Linear(linear_in*block.expansion, args.num_classes)

        self.activation = get_activation(args.activation, args, **kwargs)

    def _make_layer(self, block, planes, num_blocks, stride, nlayer, args, **kwargs):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride, nlayer, i, args, **kwargs))
            self.in_planes = planes * block.expansion
        return Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet_g(nn.Module):
    def __init__(self, block, num_blocks, args=None, **kwargs):
        super(ResNet_g, self).__init__()
        self.in_planes = 64
        if args.dataset in ["CIFAR10", "CIFAR100"]:
            linear_in = 512
        elif args.dataset == "TinyImageNet":
            linear_in = 512*2*2
        elif args.dataset == "ImageNet":
            linear_in = 512*7*7
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, nlayer=0, args=args, **kwargs)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, nlayer=1, args=args, **kwargs)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, nlayer=2, args=args, **kwargs)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, nlayer=3, args=args, **kwargs)
        self.linear = nn.Linear(linear_in*block.expansion, args.num_classes)

        self.activation = get_activation(args.activation, args, **kwargs)

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


class ResNet_hook(nn.Module):
    def __init__(self, block, num_blocks, args=None, **kwargs):
        super(ResNet_hook, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.activation = nn.ReLU()

        # Layer 1
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)

        # Layer 2
        self.conv6 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(128)
        self.shortcut1 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(128)
                )

        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(128)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(128)

        # Layer 3
        self.conv10 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(256)
        self.conv11 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(256)
        self.shortcut2 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(256)
                )

        self.conv12 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(256)
        self.conv13 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn13 = nn.BatchNorm2d(256)

        # Layer 4
        self.conv14 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn14 = nn.BatchNorm2d(512)
        self.conv15 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn15 = nn.BatchNorm2d(512)
        self.shortcut3 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(512)
                )

        self.conv16 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn16 = nn.BatchNorm2d(512)
        self.conv17 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn17 = nn.BatchNorm2d(512)

        self.linear = nn.Linear(512, args.num_classes)

        self.grads = {}
        self.get_grad = False
        self.endpoints = {}

    def set_grad(self):
        self.get_grad = True

    def save_grad(self, name):
        def hook(grad):
            self.grads[name] = grad
        return hook

    def forward(self, x):
        out1 = self.activation(self.bn1(self.conv1(x)))

        # Layer 1
        # Basic block 1
        out2 = self.activation(self.bn2(self.conv2(out1)))
        out3 = self.activation(self.bn3(self.conv3(out2)))
        # Basic block 2
        out4 = self.activation(self.bn4(self.conv4(out3)))
        out5 = self.activation(self.bn5(self.conv5(out4)))

        # Layer 2
        # Basic block 1
        out6 = self.activation(self.bn6(self.conv6(out5)))
        out7 = self.bn7(self.conv7(out6))
        out7 += self.shortcut1(out5)
        out7 = self.activation(out7)
        # Basic block 2
        out8 = self.activation(self.bn8(self.conv8(out7)))
        out9 = self.activation(self.bn9(self.conv9(out8)))

        # Layer 3
        # Basic block 1
        out10 = self.activation(self.bn10(self.conv10(out9)))
        out11 = self.bn11(self.conv11(out10))
        out11 += self.shortcut2(out9)
        out11 = self.activation(out11)
        # Basic block 2
        out12 = self.activation(self.bn12(self.conv12(out11)))
        out13 = self.activation(self.bn13(self.conv13(out12)))

        # Layer 4
        # Basic block 1
        out14 = self.activation(self.bn14(self.conv14(out13)))
        out15 = self.bn15(self.conv15(out14))
        out15 += self.shortcut3(out13)
        out15 = self.activation(out15)
        # Basic block 2
        out16 = self.activation(self.bn16(self.conv16(out15)))
        out17 = self.activation(self.bn17(self.conv17(out16)))

        out18 = F.avg_pool2d(out17, 4)
        out18 = out18.view(out18.size(0), -1)
        out18 = self.linear(out18)

        if x.requires_grad:
            x.register_hook(self.save_grad('input'))
            out1.register_hook(self.save_grad('layer1'))
            out2.register_hook(self.save_grad('layer2'))
            out3.register_hook(self.save_grad('layer3'))
            out4.register_hook(self.save_grad('layer4'))
            out5.register_hook(self.save_grad('layer5'))
            out6.register_hook(self.save_grad('layer6'))
            out7.register_hook(self.save_grad('layer7'))
            out8.register_hook(self.save_grad('layer8'))
            out9.register_hook(self.save_grad('layer9'))
            out10.register_hook(self.save_grad('layer10'))
            out11.register_hook(self.save_grad('layer11'))
            out12.register_hook(self.save_grad('layer12'))
            out13.register_hook(self.save_grad('layer13'))
            out14.register_hook(self.save_grad('layer14'))
            out15.register_hook(self.save_grad('layer15'))
            out16.register_hook(self.save_grad('layer16'))
            out17.register_hook(self.save_grad('layer17'))

            self.endpoints['input'] = x
            self.endpoints['layer1'] = out1
            self.endpoints['layer2'] = out2
            self.endpoints['layer3'] = out3
            self.endpoints['layer4'] = out4
            self.endpoints['layer5'] = out5
            self.endpoints['layer6'] = out6
            self.endpoints['layer7'] = out7
            self.endpoints['layer8'] = out8
            self.endpoints['layer9'] = out9
            self.endpoints['layer10'] = out10
            self.endpoints['layer11'] = out11
            self.endpoints['layer12'] = out12
            self.endpoints['layer13'] = out13
            self.endpoints['layer14'] = out14
            self.endpoints['layer15'] = out15
            self.endpoints['layer16'] = out16
            self.endpoints['layer17'] = out17

        return out18


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

# ResNet with grad hooks
def ResNet18_g(args, **kwargs):
    return ResNet_g(BasicBlock, [2,2,2,2], args, **kwargs)

def ResNet34_g(args, **kwargs):
    return ResNet_g(BasicBlock, [3,4,6,3], args, **kwargs)

def ResNet50_g(args, **kwargs):
    return ResNet_g(Bottleneck, [3,4,6,3], args, **kwargs)

def ResNet101_g(args, **kwargs):
    return ResNet_g(Bottleneck, [3,4,23,3], args, **kwargs)

def ResNet152_g(args, **kwargs):
    return ResNet_g(Bottleneck, [3,8,36,3], args, **kwargs)

def ResNet18_2x2(args, **kwargs):
    return ResNet_g(BasicBlock, [2,2,2,2], args, **kwargs)

def ResNet101_2x2(args, **kwargs):
    return ResNet_g(Bottleneck, [3,4,23,3], args, **kwargs)

def ResNet18_4x4(args, **kwargs):
    return ResNet_g(BasicBlock, [2,2,2,2], args, **kwargs)

def ResNet101_4x4(args, **kwargs):
    return ResNet_g(Bottleneck, [3,4,23,3], args, **kwargs)

def ResNet18_hook(args, **kwargs):
    return ResNet_hook(BasicBlock, [2,2,2,2], args, **kwargs)

def test():
    net = ResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())
