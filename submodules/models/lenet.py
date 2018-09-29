'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

from submodules.activation import get_activation, CustomNorm2d
from common.torch_utils import to_np

import os

__all__ = ['LeNet', 'LeNet_conv32', 'LeNet_conv32_nobn', 'LeNet_conv32_nobn_maxp',
           'LeNet_conv32_nobn_maxp32', 'LeNet_conv32_nobn_avgp', 'LeNet_conv32_nobn_avgp32',
           'LeNet_conv75_nobn', 'LeNet_conv42_nobn', 'LeNet_conv22_nobn']


class LeNet_conv32(nn.Module):
    def __init__(self, args, **kwargs):
        super(LeNet_conv32, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2)
        self.bn1   = CustomNorm2d(16, args, **kwargs)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2)
        self.bn2   = CustomNorm2d(32, args, **kwargs)
        self.conv3   = nn.Conv2d(32, 64, 3, stride=1)
        self.bn3   = CustomNorm2d(64, args, **kwargs)
        self.fc1   = nn.Linear(1600, 512)
        self.fc2   = nn.Linear(512, args.num_classes)

        self.activation1 = get_activation(args.activation, 'conv1', args, **kwargs)
        self.activation2 = get_activation(args.activation, 'conv2', args, **kwargs)
        self.activation3 = get_activation(args.activation, 'conv3', args, **kwargs)
        self.activation4 = get_activation(args.activation, 'fc1', args, **kwargs)

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
        conv1 = self.activation1(self.bn1(self.conv1(x)))
        conv2 = self.activation2(self.bn2(self.conv2(conv1)))
        conv3 = self.activation3(self.bn3(self.conv3(conv2)))
        fc1 = conv3.view(conv3.size(0), -1)
        fc1 = self.activation4(self.fc1(fc1))
        out = self.fc2(fc1)

        if x.requires_grad:
            x.register_hook(self.save_grad('input'))
            conv1.register_hook(self.save_grad('conv1'))
            conv2.register_hook(self.save_grad('conv2'))
            conv3.register_hook(self.save_grad('conv3'))

            self.endpoints['input'] = x
            self.endpoints['conv1'] = conv1
            self.endpoints['conv2'] = conv2
            self.endpoints['conv3'] = conv3

        return out


class LeNet_conv32_nobn(nn.Module):
    def __init__(self, args, **kwargs):
        super(LeNet_conv32_nobn, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1)

        if args.dataset in ["CIFAR10", "CIFAR100"]:
            linear_in = 1600
        elif args.dataset == "TinyImageNet":
            linear_in = 1600*2*2
        elif args.dataset == "ImageNet":
            linear_in = 1600*7*7
        else:
            raise NotImplementedError

        self.fc1   = nn.Linear(linear_in, 512)
        self.fc2   = nn.Linear(512, args.num_classes)

        self.activation1 = get_activation(args.activation, 'conv1', args, **kwargs)
        self.activation2 = get_activation(args.activation, 'conv2', args, **kwargs)
        self.activation3 = get_activation(args.activation, 'conv3', args, **kwargs)
        self.activation4 = get_activation(args.activation, 'fc1', args, **kwargs)

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
        conv1 = self.activation1(self.conv1(x))
        conv2 = self.activation2(self.conv2(conv1))
        conv3 = self.activation3(self.conv3(conv2))
        fc1 = conv3.view(conv3.size(0), -1)
        fc1 = self.activation4(self.fc1(fc1))
        out = self.fc2(fc1)

        if x.requires_grad:
            x.register_hook(self.save_grad('input'))
            conv1.register_hook(self.save_grad('conv1'))
            conv2.register_hook(self.save_grad('conv2'))
            conv3.register_hook(self.save_grad('conv3'))

            self.endpoints['input'] = x
            self.endpoints['conv1'] = conv1
            self.endpoints['conv2'] = conv2
            self.endpoints['conv3'] = conv3

        return out


class LeNet_conv32_nobn_maxp(nn.Module):
    def __init__(self, args, **kwargs):
        super(LeNet_conv32_nobn_maxp, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1)
        self.fc1   = nn.Linear(64*4*4, 512)
        self.fc2   = nn.Linear(512, args.num_classes)

        self.activation1 = get_activation(args.activation, 'conv1', args, **kwargs)
        self.activation2 = get_activation(args.activation, 'conv2', args, **kwargs)
        self.activation3 = get_activation(args.activation, 'conv3', args, **kwargs)
        self.activation4 = get_activation(args.activation, 'fc1', args, **kwargs)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

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
        conv1 = self.maxpool(self.activation1(self.conv1(x)))
        conv2 = self.maxpool(self.activation2(self.conv2(conv1)))
        conv3 = self.activation3(self.conv3(conv2))
        fc1 = conv3.view(conv3.size(0), -1)
        fc1 = self.activation4(self.fc1(fc1))
        out = self.fc2(fc1)

        if x.requires_grad:
            x.register_hook(self.save_grad('input'))
            conv1.register_hook(self.save_grad('conv1'))
            conv2.register_hook(self.save_grad('conv2'))
            conv3.register_hook(self.save_grad('conv3'))

            self.endpoints['input'] = x
            self.endpoints['conv1'] = conv1
            self.endpoints['conv2'] = conv2
            self.endpoints['conv3'] = conv3

        return out


class LeNet_conv32_nobn_maxp32(nn.Module):
    def __init__(self, args, **kwargs):
        super(LeNet_conv32_nobn_maxp32, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1)
        self.fc1   = nn.Linear(64*3*3, 512)
        self.fc2   = nn.Linear(512, args.num_classes)

        self.activation1 = get_activation(args.activation, 'conv1', args, **kwargs)
        self.activation2 = get_activation(args.activation, 'conv2', args, **kwargs)
        self.activation3 = get_activation(args.activation, 'conv3', args, **kwargs)
        self.activation4 = get_activation(args.activation, 'fc1', args, **kwargs)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

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
        conv1 = self.maxpool(self.activation1(self.conv1(x)))
        conv2 = self.maxpool(self.activation2(self.conv2(conv1)))
        conv3 = self.activation3(self.conv3(conv2))
        fc1 = conv3.view(conv3.size(0), -1)
        fc1 = self.activation4(self.fc1(fc1))
        out = self.fc2(fc1)

        if x.requires_grad:
            x.register_hook(self.save_grad('input'))
            conv1.register_hook(self.save_grad('conv1'))
            conv2.register_hook(self.save_grad('conv2'))
            conv3.register_hook(self.save_grad('conv3'))

            self.endpoints['input'] = x
            self.endpoints['conv1'] = conv1
            self.endpoints['conv2'] = conv2
            self.endpoints['conv3'] = conv3

        return out


class LeNet_conv32_nobn_avgp(nn.Module):
    def __init__(self, args, **kwargs):
        super(LeNet_conv32_nobn_avgp, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1)
        self.fc1   = nn.Linear(64*4*4, 512)  # For pool 2x2 stride2
        #self.fc1   = nn.Linear(64*3*3, 512)  # For pool 3x3 stride2
        self.fc2   = nn.Linear(512, args.num_classes)

        self.activation1 = get_activation(args.activation, 'conv1', args, **kwargs)
        self.activation2 = get_activation(args.activation, 'conv2', args, **kwargs)
        self.activation3 = get_activation(args.activation, 'conv3', args, **kwargs)
        self.activation4 = get_activation(args.activation, 'fc1', args, **kwargs)

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

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
        conv1 = self.avgpool(self.activation1(self.conv1(x)))
        conv2 = self.avgpool(self.activation2(self.conv2(conv1)))
        conv3 = self.activation3(self.conv3(conv2))
        fc1 = conv3.view(conv3.size(0), -1)
        fc1 = self.activation4(self.fc1(fc1))
        out = self.fc2(fc1)

        if x.requires_grad:
            x.register_hook(self.save_grad('input'))
            conv1.register_hook(self.save_grad('conv1'))
            conv2.register_hook(self.save_grad('conv2'))
            conv3.register_hook(self.save_grad('conv3'))

            self.endpoints['input'] = x
            self.endpoints['conv1'] = conv1
            self.endpoints['conv2'] = conv2
            self.endpoints['conv3'] = conv3

        return out

class LeNet_conv32_nobn_avgp32(nn.Module):
    def __init__(self, args, **kwargs):
        super(LeNet_conv32_nobn_avgp32, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1)
        self.fc1   = nn.Linear(64*3*3, 512)
        self.fc2   = nn.Linear(512, args.num_classes)

        self.activation1 = get_activation(args.activation, 'conv1', args, **kwargs)
        self.activation2 = get_activation(args.activation, 'conv2', args, **kwargs)
        self.activation3 = get_activation(args.activation, 'conv3', args, **kwargs)
        self.activation4 = get_activation(args.activation, 'fc1', args, **kwargs)

        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2)

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
        conv1 = self.avgpool(self.activation1(self.conv1(x)))
        conv2 = self.avgpool(self.activation2(self.conv2(conv1)))
        conv3 = self.activation3(self.conv3(conv2))
        fc1 = conv3.view(conv3.size(0), -1)
        fc1 = self.activation4(self.fc1(fc1))
        out = self.fc2(fc1)

        if x.requires_grad:
            x.register_hook(self.save_grad('input'))
            conv1.register_hook(self.save_grad('conv1'))
            conv2.register_hook(self.save_grad('conv2'))
            conv3.register_hook(self.save_grad('conv3'))

            self.endpoints['input'] = x
            self.endpoints['conv1'] = conv1
            self.endpoints['conv2'] = conv2
            self.endpoints['conv3'] = conv3

        return out

class LeNet_conv75_nobn(nn.Module):
    def __init__(self, args, **kwargs):
        super(LeNet_conv75_nobn, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 7, stride=5)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1)
        self.fc1   = nn.Linear(256, 512)
        self.fc2   = nn.Linear(512, args.num_classes)

        self.activation1 = get_activation(args.activation, 'conv1', args, **kwargs)
        self.activation2 = get_activation(args.activation, 'conv2', args, **kwargs)
        self.activation3 = get_activation(args.activation, 'conv3', args, **kwargs)
        self.activation4 = get_activation(args.activation, 'fc1', args, **kwargs)

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
        conv1 = self.activation1(self.conv1(x))
        conv2 = self.activation2(self.conv2(conv1))
        conv3 = self.activation3(self.conv3(conv2))
        fc1 = conv3.view(conv3.size(0), -1)
        fc1 = self.activation4(self.fc1(fc1))
        out = self.fc2(fc1)

        if x.requires_grad:
            x.register_hook(self.save_grad('input'))
            conv1.register_hook(self.save_grad('conv1'))
            conv2.register_hook(self.save_grad('conv2'))
            conv3.register_hook(self.save_grad('conv3'))

            self.endpoints['input'] = x
            self.endpoints['conv1'] = conv1
            self.endpoints['conv2'] = conv2
            self.endpoints['conv3'] = conv3

        return out


class LeNet(nn.Module):
    def __init__(self, args, **kwargs):
        super(LeNet, self).__init__()
        #self.conv1 = nn.Conv2d(1, 6, 5)  # For mnist
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1   = CustomNorm2d(6, args, **kwargs)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2   = CustomNorm2d(16, args, **kwargs)

        if args.dataset in ["CIFAR10", "CIFAR100"]:
            linear_in = 400
        elif args.dataset == "MNIST":
            linear_in = 256
        elif args.dataset == "ImageNet":
            linear_in = 44944
        else:
            raise NotImplementedError

        self.fc1   = nn.Linear(linear_in, 120)
        self.bn3   = CustomNorm2d(120, args, **kwargs)
        self.fc2   = nn.Linear(120, 84)
        self.bn4   = CustomNorm2d(84, args, **kwargs)
        self.fc3   = nn.Linear(84, args.num_classes)

        self.activation1 = get_activation(args.activation, 'conv1', args, **kwargs)
        self.activation2 = get_activation(args.activation, 'conv2', args, **kwargs)
        self.activation3 = get_activation(args.activation, 'fc1', args, **kwargs)
        self.activation4 = get_activation(args.activation, 'fc2', args, **kwargs)

    def forward(self, x):
        out = self.activation1(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, 2)
        out = self.activation2(self.bn2(self.conv2(out)))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.activation3(self.bn3(self.fc1(out)))
        out = self.activation4(self.bn4(self.fc2(out)))
        out = self.fc3(out)
        return out


class LeNet_layerwise(nn.Module):
    def __init__(self, args, **kwargs):
        super(LeNet_layerwise, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1   = CustomNorm2d(6, args, **kwargs)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2   = CustomNorm2d(16, args, **kwargs)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.bn3   = CustomNorm2d(120, args, **kwargs)
        self.fc2   = nn.Linear(120, 84)
        self.bn4   = CustomNorm2d(84, args, **kwargs)
        self.fc3   = nn.Linear(84, args.num_classes)

        self.activation1 = get_activation(args.activation, 'conv1', args, **kwargs)
        self.activation2 = get_activation(args.activation, 'conv2', args, **kwargs)
        self.activation3 = get_activation(args.activation, 'fc1', args, **kwargs)
        self.activation4 = get_activation(args.activation, 'fc2', args, **kwargs)

        self.fc_out1 = nn.Linear(6*28*28, args.num_classes)
        self.fc_out2 = nn.Linear(16*10*10, args.num_classes)

    def forward(self, x):
        end_points = {}
        out = self.activation1(self.bn1(self.conv1(x)))
        end_points['conv1'] = self.fc_out1(out.view(out.size(0), -1))

        out = F.max_pool2d(out, 2)
        out = self.activation2(self.bn2(self.conv2(out)))
        end_points['conv2'] = self.fc_out2(out.view(out.size(0), -1))

        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.activation3(self.bn3(self.fc1(out)))
        out = self.activation4(self.bn4(self.fc2(out)))
        out = self.fc3(out)
        end_points['out'] = out
        return out, end_points


class LeNet_conv42_nobn(nn.Module):
    def __init__(self, args, **kwargs):
        super(LeNet_conv42_nobn, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 4, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=2)
        self.fc1   = nn.Linear(64*2*2, 512)
        self.fc2   = nn.Linear(512, args.num_classes)

        self.activation1 = get_activation(args.activation, 'conv1', args, **kwargs)
        self.activation2 = get_activation(args.activation, 'conv2', args, **kwargs)
        self.activation3 = get_activation(args.activation, 'conv3', args, **kwargs)
        self.activation4 = get_activation(args.activation, 'fc1', args, **kwargs)

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
        conv1 = self.activation1(self.conv1(x))
        conv2 = self.activation2(self.conv2(conv1))
        conv3 = self.activation3(self.conv3(conv2))
        fc1 = conv3.view(conv3.size(0), -1)
        fc1 = self.activation4(self.fc1(fc1))
        out = self.fc2(fc1)

        if x.requires_grad:
            x.register_hook(self.save_grad('input'))
            conv1.register_hook(self.save_grad('conv1'))
            conv2.register_hook(self.save_grad('conv2'))
            conv3.register_hook(self.save_grad('conv3'))

            self.endpoints['input'] = x
            self.endpoints['conv1'] = conv1
            self.endpoints['conv2'] = conv2
            self.endpoints['conv3'] = conv3

        return out


class LeNet_conv22_nobn(nn.Module):
    def __init__(self, args, **kwargs):
        super(LeNet_conv22_nobn, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 2, stride=2)
        self.fc1   = nn.Linear(64*4*4, 512)
        self.fc2   = nn.Linear(512, args.num_classes)

        self.activation1 = get_activation(args.activation, 'conv1', args, **kwargs)
        self.activation2 = get_activation(args.activation, 'conv2', args, **kwargs)
        self.activation3 = get_activation(args.activation, 'conv3', args, **kwargs)
        self.activation4 = get_activation(args.activation, 'fc1', args, **kwargs)

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
        conv1 = self.activation1(self.conv1(x))
        conv2 = self.activation2(self.conv2(conv1))
        conv3 = self.activation3(self.conv3(conv2))
        fc1 = conv3.view(conv3.size(0), -1)
        fc1 = self.activation4(self.fc1(fc1))
        out = self.fc2(fc1)

        if x.requires_grad:
            x.register_hook(self.save_grad('input'))
            conv1.register_hook(self.save_grad('conv1'))
            conv2.register_hook(self.save_grad('conv2'))
            conv3.register_hook(self.save_grad('conv3'))

            self.endpoints['input'] = x
            self.endpoints['conv1'] = conv1
            self.endpoints['conv2'] = conv2
            self.endpoints['conv3'] = conv3

        return out
