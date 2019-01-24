import numpy as np
import torch.nn as nn

from common.torch_utils import get_model

__all__ = ['ace', 'ace_resnet101']


class ACE(nn.Module):
    """Induce and evade adversarial attacks by the checkerboard artifact"""
    def __init__(self, classifiers, autoencoders, stacks, lambdas, shifts,
                 args, **kwargs):
        super(ACE, self).__init__()
        self.args = args

        # Load all possible combinations
        self.classifiers = nn.ModuleList([])
        self.autoencoders = nn.ModuleList([])

        # Classifiers are always frozen
        for c in classifiers:
            args.model = c
            model, _ = get_model(args)
            for param in model.parameters():
                param.requires_grad = False
            self.classifiers.append(model)

        self.stacks = stacks

        # Autoencoders can be trained
        for a in autoencoders:
            args.model = a
            model, _ = get_model(args)
            self.autoencoders.append(model)

        self.lambdas = lambdas
        self.shifts = shifts

    def forward(self, x):
        classifier = np.random.choice(self.classifiers)
        stacks = np.random.choice(self.stacks)
        autoencoders = np.random.choice(self.autoencoders, stacks)
        lambdas = np.random.choice(self.lambdas, stacks)

        shifts = []
        for _ in range(stacks):
            idx = np.random.choice(len(self.shifts))
            shifts.append(self.shifts[idx])

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


# Default configurations
classifiers = ['densenet121', 'resnet101', 'vgg19', 'vgg19_bn']
autoencoders = ['unet']
stacks = [1]
lambdas = [0, 0.5, 0.7, 0.9, 0.99, 1]
shifts = [(0,0), (0,1), (1,0), (0,-1), (-1,0), (1,1), (-1, -1), (1,-1), (-1,1)]


def ace(args, **kwargs):
    assert args.dataset == "ImageNet"
    global classifiers, autoencoders, stacks, lambdas, shifts
    return ACE(classifiers, autoencoders, stacks, lambdas, shifts, args, **kwargs)

def ace_resnet101(args, **kwargs):
    assert args.dataset == "ImageNet"
    global autoencoders, stacks
    classifiers = ['resnet101']
    lamdas = [1]
    shifts = [(0,0)]
    return ACE(classifiers, autoencoders, stacks, lambdas, shifts, args, **kwargs)
