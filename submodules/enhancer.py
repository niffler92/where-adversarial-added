from collections import Iterable
import torch
import torch.nn as nn

from common.torch_utils import to_var, get_model
from submodules.models import *
from submodules.autoencoders import *

__all__ = ['ResNet18_en', 'Vgg11_en', 'ResNet152_en', 'Vgg19_en', 'LeNet_en']
__all__ += ['segnet_resnet152', 'segnet_vgg19']
__all__ += ['unet_resnet152', 'unet_vgg19']


class Enhancer(nn.Module):
    def __init__(self, *conv, lambd=1):
        super(Enhancer, self).__init__()
        assert len(conv) == 4
        in_channels, out_channels, kernel_size, stride = conv
        self.lambd = lambd

        self.init = True
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride),
            nn.ConvTranspose2d(out_channels, in_channels, kernel_size, stride)
        )

    def forward(self, x):
        if self.init:
            self.init_shape(x)
        out = x + self.lambd*self.padding(self.shortcut(x))
        return out

    def init_shape(self, x):
        out = self.shortcut(x)
        x_W, x_H = x.shape[2:]
        out_W, out_H = out.shape[2:]
        self.padding = nn.ZeroPad2d((0, x_W - out_W, 0, x_H - out_H))
        self.init = False


class EnNet(nn.Module):
    def __init__(self, net, enhancer, args, **kwargs):
        super(EnNet, self).__init__()
        self.net = net
        self.enhancer = enhancer
        self.args = args

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
        if hasattr(x, 'requires_grad') and x.requires_grad:
            x.register_hook(self.save_grad('input'))
            self.endpoints['input'] = x

        x = self.enhancer(x)
        out = self.net(x)

        return out


def ResNet18_en(args, **kwargs):
    return EnNet(ResNet18(args, **kwargs), Enhancer(3, 3, 3, 2, lambd=args.lambd), args, **kwargs)

def Vgg11_en(args, **kwargs):
    return EnNet(Vgg11(args, **kwargs), Enhancer(3, 3, 3, 2, lambd=args.lambd), args, **kwargs)

def ResNet152_en(args, **kwargs):
    return EnNet(ResNet152(args, **kwargs), Enhancer(3,3,3,2,lambd=args.lambd), args, **kwargs)

def Vgg19_en(args, **kwargs):
    return EnNet(Vgg19(args, **kwargs), Enhancer(3,3,3,2,lambd=args.lambd), args, **kwargs)

def LeNet_en(args, **kwargs):
    return EnNet(LeNet(args, **kwargs), Enhancer(3,3,3,2,lambd=args.lambd), args, **kwargs)

# Enhanced Autoencoder Networks
class AENet(nn.Module):
    def __init__(self, autoencoder, net, args, **kwargs):
        super().__init__()
        self.autoencoder = autoencoder
        self.net = net
        self.args = args

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
        if hasattr(x, 'requires_grad') and x.requires_grad:
            x.register_hook(self.save_grad('input'))
            self.endpoints['input'] = x
        out = self.autoencoder(x)
        out = (1 - self.args.lambd)*x + self.args.lambd*out
        out = self.net(out)
        return out


def get_ae_model(autoencoder, model, args, **kwargs):
    multigpu = args.multigpu
    args.multigpu = 0
    args.model = model
    net = get_model(args)
    args.model = autoencoder
    args.ckpt_name = args.ckpt_ae
    autoencoder = get_model(args)
    aenet = AENet(autoencoder, net, args, **kwargs)
    if multigpu:
        aenet = nn.DataParallel(aenet, device_ids=list(range(multigpu)))
    return aenet


def segnet_resnet152(args, **kwargs):
    if args.pretrained:
        return get_ae_model('segnet', 'resnet152', args, **kwargs)
    else:
        return AENet(segnet(args, **kwargs), ResNet152(args, **kwargs), args, **kwargs)

def segnet_vgg19(args, **kwargs):
    if args.pretrained:
        return get_ae_model('segnet', 'vgg19', args, **kwargs)
    else:
        return AENet(segnet(args, **kwargs), Vgg19(args, **kwargs), args, **kwargs)

def unet_resnet152(args, **kwargs):
    if args.pretrained:
        return get_ae_model('unet', 'resnet152', args, **kwargs)
    else:
        return AENet(unet(args, **kwargs), ResNet152(args, **kwargs), args, **kwargs)

def unet_vgg19(args, **kwargs):
    if args.pretrained:
        return get_ae_model('unet', 'vgg19', args, **kwargs)
    else:
        return AENet(unet(args, **kwargs), Vgg19(args, **kwargs), args, **kwargs)
