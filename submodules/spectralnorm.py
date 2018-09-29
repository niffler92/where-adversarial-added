import torch
import torch.nn as nn
from torch.autograd import Variable


class SpectralNorm(nn.Module):
    """Spectral normalization of weight with power iteration
    """
    def __init__(self, module, do_sn, niter=1):
        super().__init__()
        self.module = module
        self.do_sn = do_sn
        self.niter = niter

        self.init_params(module)

    @staticmethod
    def init_params(module):
        """u, v, W_sn
        """
        w = module.weight
        height = w.size(0)
        width = w.view(w.size(0), -1).shape[-1]

        u = torch.randn(height, 1)
        v = torch.randn(1, width)
        module.register_buffer('u', u)
        module.register_buffer('v', v)

    @staticmethod
    def update_params(module, niter):
        u, v, w = module.u, module.v, module.weight
        height = w.size(0)

        for i in range(niter):  # Power iteration
            v = w.view(height, -1).t().data @ u
            v /= (v.norm(p=2) + 1e-12)
            u = w.view(height, -1).data @ v
            u /= (u.norm(p=2) + 1e-12)

        w.data /= (u.t() @ w.view(height, -1).data @ v)  # Spectral normalization

    def forward(self, x):
        if self.do_sn:
            self.update_params(self.module, self.niter)
        return self.module(x)
