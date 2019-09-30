import numpy as np
from scipy import stats
import torch

from common.torch_utils import to_np, to_var

__all__ = ['regiondefense']


class RegionDefense:
    def __init__(self, model, r=0.02, m=1000, args=None, **kwargs):
        self.model = model
        self.r = r
        self.m = m
        self.args = args

    def generate(self, images, labels):
        ensemble = []
        for _ in range(self.m):
            noise = 2*self.r*(torch.rand(*images.shape) - 0.5)
            if self.args.cuda:
                noise = noise.cuda()
            def_outputs = self.model(to_var(images + noise))
            ensemble.append(to_np(torch.max(def_outputs, 1)[1]))
        ensemble = np.asarray(ensemble)
        def_labels, _ = stats.mode(ensemble)

        def_images = images
        def_labels = torch.LongTensor(def_labels).squeeze(0)
        if self.args.cuda:
            def_labels = def_labels.cuda()

        return def_images, def_labels


def regiondefense(model, args, **kwargs):
    return RegionDefense(model, args=args, **kwargs)
