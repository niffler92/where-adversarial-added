import numpy as np
from scipy import stats
import torch

from common.torch_utils import to_np, to_var
from submodules.defenses.pixelshift import PixelShift

__all__ = ['pixelensemble']


class PixelEnsemble:
    def __init__(self, model, delta=1.5, pad_type="zero", nensemble=9, args=None, **kwargs):
        assert delta > 0, "delta must be positive"
        assert pad_type in ["reflection", "replication", "zero"]
        self.args = args
        self.ensemble = []
        # for x in range(-int(delta), int(delta)+1):
            # for y in range(-int(delta), int(delta)+1):
                # if x == 0 and y == 0:
                    # continue
                # if x**2 + y**2 <= delta**2:
                    # ps = PixelShift(model, x, y, pad_type, args=args, **kwargs)
                    # self.ensemble.append(ps)

        choices = [(0, 0), (1, 0), (0, 1), (1, 1), (-1, 0), (0, -1), (1, -1), (-1, 1), (-1, -1)]
        for i in range(nensemble):
            # x, y = choices[np.random.choice(range(len(choices)), size=1)[0]]
            x, y = choices[i]
            ps = PixelShift(model, x, y, pad_type, args=args, **kwargs)
            self.ensemble.append(ps)


    def generate(self, images, labels):
        ensemble = []
        for ps in self.ensemble:
            _, def_labels = ps.generate(images, labels)
            ensemble.append(to_np(def_labels))
        ensemble = np.asarray(ensemble)
        def_labels, _ = stats.mode(ensemble)

        def_images = images
        def_labels = torch.LongTensor(def_labels)

        if torch.cuda.is_available():
            def_labels = def_labels.cuda()

        return def_images, def_labels


def pixelensemble(model, args, **kwargs):
    return PixelEnsemble(model, args=args, **kwargs)
