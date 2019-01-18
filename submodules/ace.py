import numpy as np
import torch.nn as nn

from common.torch_utils import get_model

__all__ = ['ace']


class ACE(nn.Module):
    """Induce and evade adversarial attacks by the checkerboard artifact"""
    def __init__(self, classifiers, autoencoders, stacks, lambdas, shifts,
                 args, **kwargs):
        super(ACE, self).__init__()
        self.id = args.id
        self.args = args
        args.pretrained = True

        self.classifiers = nn.ModuleList([])
        self.autoencoders = nn.ModuleList([])

        # Load all possible combinations
        if self.id == 0:
            for c in classifiers:
                args.model = c
                self.classifiers.append(get_model(args)[0])
            
            self.stacks = stacks

            for a in autoencoders:
                args.model = a
                self.autoencoders.append(get_model(args)[0])

            self.lambdas = lambdas
            self.shifts = shifts

        # Select configuration by modulo
        else:
            args.model = classifiers[self.id % len(classifiers)]
            self.classifiers.append(get_model(args)[0])

            self.stacks = stacks[self.id % len(stacks)]

            autoencoders = self.get_config(autoencoders)
            for a in autoencoders:
                args.model = a
                self.autoencoders.append(get_model(args)[0])
            
            self.lambdas = self.get_config(lambdas)
            self.shifts = self.get_config(shifts)

    def forward(self, x):
        if self.id == 0:
            classifier = np.random.choice(self.classifiers)
            stacks = np.random.choice(self.stacks)
            autoencoders = np.random.choice(self.autoencoders, stacks)
            lambdas = np.random.choice(self.lambdas, stacks)
            shifts = np.random.choice(self.shifts, stacks)

        else:
            classifier = self.classifiers[0]
            autoencoders = self.autoencoders
            lambdas = self.lambdas
            shifts = self.shifts

        for idx, autoencoder in enumerate(autoencoders):           
            l, r, u, d = self.pad_shape(*shifts[idx])
            x_pad = nn.ZeroPad2d(padding=(l,r,u,d))(x)
            w, h = x_pad.shape[-1], x_pad.shape[-2]
            
            x_shift = x_pad[:,:,d:h-u,r:w-l].contiguous()
            x = lambdas[idx]*autoencoder(x_shift) + (1 - lambdas[idx])*x_shift
        
        out = classifier(x)

        return out

    def pad_shape(self, x, y):
        left = x*(x > 0)
        right = -x*(x < 0)
        up = y*(y > 0)
        down = -y*(y < 0)

        return left, right, up, down
    
    def get_config(self, options):
        config = []
        base = len(options)
        idx = self.id % (base**self.stacks)
        while idx > 0:
            config.append(options[idx % base])
            idx /= base
        
        return config


def ace(args, **kwargs):
    assert args.dataset == "ImageNet"
    
    classifiers = [
            'alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201',
            'inception_v3', 'resnet101', 'resnet152', 'resnet18', 'resnet34',
            'resnet50', 'squeezenet1_0', 'squeezenet1_1', 'vgg11', 'vgg13',
            'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn',
    ]
    autoencoders = ['unet']
    stacks = [1]
    lambdas = [0.5]
    shifts = [(0,1)]

    return ACE(classifiers, autoencoders, stacks, lambdas, shifts, args, **kwargs)
