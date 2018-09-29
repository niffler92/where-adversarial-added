import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from scipy.stats import norm

from common.torch_utils import to_np


class CustomNorm2d(nn.Module):
    def __init__(self, num_features, args=None, **kwargs):
        super(CustomNorm2d, self).__init__()
        if 'normalization' in kwargs:
            self.normalization = kwargs.get('normalization')
        else:
            self.normalization = args.normalization

        if args.no_bn:
            self.norm = nn.Dropout2d(0)
        else:
            if self.normalization == 'batch':
                self.norm = nn.BatchNorm2d(num_features, affine=True)
            elif self.normalization == 'instance':
                self.norm = nn.InstanceNorm2d(num_features, affine=True)
        if args.no_affine:
            self.norm.register_parameter('weight', None)
            self.norm.register_parameter('bias', None)
        if args.half:
            self.norm.float()

    def forward(self, x):
        return self.norm(x)


class CustomActivation(nn.Module):
    def __init__(self, name=None, args=None, **kwargs):
        super(CustomActivation, self).__init__()
        self.args = args
        self.name = name
        self.info = {}

    def record(self, out):
        if self.args.record:
            nonzero = to_np(out)
            nonzero[nonzero==0] = np.inf
            self.info['threshold'] = float(np.min(nonzero))
            self.info['activation_ratio'] = to_np((out>0)+(out<0)).mean()
            self.info['active_nodes'] = to_np((out>0)+(out<0)).sum()/out.size(0)


class AeLU(CustomActivation):
    def __init__(self, keep=0.99, inplace=False, random=False, name=None, args=None, **kwargs):
        super(AeLU, self).__init__(name=name, args=args, **kwargs)
        self.keep = keep if not random else np.random.uniform(0.25, 0.75)
        self.inplace = inplace
        self.grad = 0  # must create backward_keep to update

    def forward(self, x):
        # FIXME: takes too long to compute
        """
        dim = list(x.size())
        x = x.view(x.size(0),-1)
        k = int(np.prod(dim[1:])*(self.keep))
        topk = to_np(x)
        topk.sort()
        topk = topk[:,-k]
        out = torch.zeros_like(x)
        for b in range(dim[0]):
            out[b] = F.threshold(x[b], float(topk[b]), 0)
        out = out.view(dim)
        return out
        """
        mean = float(to_np(x.mean()))
        std = float(to_np(x.std()))
        threshold = norm.ppf(1-self.keep, loc=mean, scale=std)
        out = F.threshold(x, threshold, 0, self.inplace)
        self.record(out)
        return out


class AeLU_T(CustomActivation):
    """
    Args:
        T (float): Our threshold value to be learned
    """
    def __init__(self, T=0, name=None, args=None, **kwargs):
        super(AeLU_T, self).__init__(name=name, args=args, **kwargs)
        self.T = nn.Parameter(torch.Tensor([T]))

    def forward(self, x):
        mean = float(to_np(x.mean()))
        std = float(to_np(x.std()))
        threshold = self.T*std + mean
        out = x * (x > threshold).float()
        self.record(out)
        return out


class Swish(CustomActivation):
    def __init__(self, name=None, args=None, **kwargs):
        super(Swish, self).__init__(name=name, args=args, **kwargs)

    def forward(self, x):
        out = torch.mul(x, torch.sigmoid(x))
        self.record(out)
        return out


class ReLU(CustomActivation):
    def __init__(self, name=None, args=None, **kwargs):
        super(ReLU, self).__init__(name=name, args=args, **kwargs)

    def forward(self, x):
        out = F.relu(x)
        self.record(out)
        return out


class Sigmoid(CustomActivation):
    def __init__(self, name=None, args=None, **kwargs):
        super(Sigmoid, self).__init__(name=name, args=args, **kwargs)

    def forward(self, x):
        out = F.sigmoid(x)
        self.record(out)
        return out


def get_activation(fn, name=None, args=None, **kwargs):
    fn = fn.lower()

    # CustomActivations
    if fn == "sigmoid":
        return Sigmoid(name=name, args=args, **kwargs)
    elif fn == "swish":
        return Swish(name=name, args=args, **kwargs)
    elif fn == "relu":
        return ReLU(name=name, args=args, **kwargs)
    elif fn == 'aelu':
        return AeLU(name=name, args=args, **kwargs)
    elif fn == 'aelu_rand':
        return AeLU(random=True, name=name, args=args, **kwargs)
    elif fn == 'aelu_t':
        return AeLU_T(name=name, args=args, **kwargs)

    # Built-in Activations
    elif fn == "relu6":
        return nn.ReLU6()
    elif fn == "selu":
        return nn.SELU()
    elif fn == "elu":
        return nn.ELU()
    elif fn == "leakyrelu":
        return nn.LeakyReLU()
    else:
        raise ValueError("Invalid activation name")
