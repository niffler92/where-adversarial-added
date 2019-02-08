import numpy as np
from scipy import stats
import torch

from common.torch_utils import one_hot

__all__ = ['regiondefense']


class RegionDefense:
    def __init__(self, model, r=0.02, m=1000, args=None, **kwargs):
        self.model = model
        self.r = r
        self.m = m
        self.args = args

    def generate(self, images):
        ensemble = []
        for _ in range(self.m):
            noise = 2*self.r*(torch.rand(*images.shape) - 0.5)
            if self.args.cuda:
                noise = noise.cuda()
            def_outputs = self.model(images + noise)
            ensemble.append(torch.max(def_outputs, 1)[1].detach().cpu().numpy())
        ensemble = np.asarray(ensemble)
        def_labels, _ = stats.mode(ensemble)

        def_labels = torch.LongTensor(def_labels).squeeze(0)
        def_labels = torch.cat([one_hot(label, self.args.num_classes).view(1,-1) for label in def_labels])
        if self.args.cuda:
            def_labels = def_labels.cuda()

        return def_labels


def regiondefense(model, args, **kwargs):
    return RegionDefense(model, args=args, **kwargs)
