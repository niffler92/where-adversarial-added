from collections import Iterable
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from differential_evolution import differential_evolution
from dataloader import get_loader, normalize, denormalize
from common.torch_utils import to_np, to_var
from common.attack_utils import get_artifact
from common.summary import AverageMeter


__all__ = ['onepixel', 'onepixel_latin', 'onepixel_64']


class OnePixel:
    def __init__(self, model, n_pix=1, strategy='rand1bin', max_iter=100, popsize=400,
                 init='normal', target=None, args=None, **kwargs):
        """An Attacker algorithm must have a generate method
        """
        self.model = model
        self.n_pix = n_pix
        self.strategy = strategy
        self.max_iter = max_iter
        self.popsize = popsize
        self.init = init
        self.target = target
        self.args = args
        self.kwargs = kwargs

        self.step_meter = AverageMeter()


    def generate(self, images, labels):
        """Generate adversarial images
        """
        preds = np.argmax(to_np(self.model(to_var(images))), axis=1)
        images = denormalize(images, self.args.dataset)*255
        #self.n_pix = int(images.size(2)*images.size(3)*self.args.gamma)

        bounds = [(0, images[0].size(1)-1), (0, images[0].size(2)-1),
                  (0, 255), (0, 255), (0, 255)] * self.n_pix

        adv_images = []
        adv_labels = []

        for i in range(len(images)):
            self.image = images[i]
            self.label = int(preds[i])

            if self.target is not None:
                self.label = self.target
            self.convergence = False

            if self.init == 'normal':
                x_loc = np.random.uniform(0, images[0].size(1), self.n_pix*self.popsize)
                y_loc = np.random.uniform(0, images[0].size(2), self.n_pix*self.popsize)
                val = np.array(np.split(np.random.normal(128, 127, self.n_pix*self.popsize*3),3))
                init = np.array(np.split(np.vstack((x_loc, y_loc, val)), self.n_pix, axis=1))
                init = np.transpose(init.reshape(-1, self.popsize))

            else:
                init = self.init

            self.step = 0
            if self.args.domain_restrict:
                self.mapping = self.create_map(self.args.gamma, self.kwargs.get('artifact'))
            else:
                self.mapping = lambda x,y: (x,y)

            result = differential_evolution(self.optimize, bounds,
                                            init=init,
                                            strategy=self.strategy,
                                            maxiter=self.max_iter,
                                            popsize=self.popsize,
                                            seed=self.args.seed,
                                            callback=self.callback,
                                            mutation=0.5,
                                            recombination=1,
                                            polish=False,
                                            tol=0,
                                            atol=-1)

            adv_image = self.perturb(result.x).squeeze(0)
            adv_images.append(adv_image)
            adv_labels.append(self.label)

            self.step_meter.update(self.step-1)

        #print("Average step per iter: {}".format(self.step_meter.avg))

        return torch.stack(adv_images), torch.LongTensor(adv_labels)#, torch.FloatTensor(steps)

    def perturb(self, x):
        adv_images = self.image.clone().unsqueeze(0)
        pixs = np.transpose(x)
        pixs = np.array(np.split(pixs, self.n_pix))

        if pixs.ndim == 2:
            (dim1, dim2) = pixs.shape
            pixs = pixs.reshape(dim1, dim2, 1)
        else:
            adv_images = adv_images.repeat(self.popsize,1,1,1)

        for i, img in enumerate(adv_images):
            for pix in pixs:
                loc = torch.LongTensor(pix[:2])
                val = torch.LongTensor(pix[2:])
                if self.args.cuda:
                    loc = loc.cuda()
                    val = val.cuda()
                x = loc[0,i]
                y = loc[1,i]
                x, y = self.mapping(x, y)
                img[:,x,y] = val[:,i]

        adv_images = normalize(adv_images/255, self.args.dataset)

        return adv_images

    def optimize(self, x):
        self.step += 1
        adv_images = Variable(self.perturb(x))
        out = self.model(adv_images)
        out = to_np(F.softmax(out, dim=1))

        if self.target is not None:
            labels = np.argmax(out, axis=1)
            if any(labels == np.ones_like(labels)*self.label):
                self.convergence = True
        else:
            labels = np.argmax(out, axis=1)
            if any(labels != np.ones_like(labels)*self.label):
                self.convergence = True

        out = out[:,self.label]
        if self.target is not None:
            out = -out
        return out

    def callback(self, x, convergence):
        return self.convergence

    def create_map(self, gamma, artifact):
        if artifact is None:
            artifact = torch.ones(self.image.size(1), self.image.size(2))
        mask = artifact

        orig_size = np.prod(artifact.shape)
        mask_size = torch.sum(mask.float())

        """
        self.ratio = gamma
        if self.ratio is not None:
            remove_pix = int(round(np.clip(mask_size - orig_size*self.ratio, 0, None)))
            coords = np.where(mask)
            remove_coords = np.random.choice(range(len(coords[0])), remove_pix, replace=False)
            for coord in remove_coords:
                x = coords[0][coord]
                y = coords[1][coord]
                mask[x][y] = False
        """

        mask_size = torch.sum(mask.float())
        self.ratio = np.sqrt(mask_size/orig_size)

        mask_x = int(artifact.size(0)*self.ratio)
        mask_y = int(artifact.size(1)*self.ratio)
        self.mask = np.empty((mask_x, mask_y), dtype=tuple)

        def mapping(x, y):
            x = np.clip(int(x*self.ratio), 0, self.mask.shape[0] - 1)
            y = np.clip(int(y*self.ratio), 0, self.mask.shape[1] - 1)
            return self.mask[x][y]

        x, y = 0, 0
        for i in range(mask.size(0)):
            for j in range(mask.size(1)):
                if mask[i][j]:
                    self.mask[x][y] = (i, j)
                    x += 1
                    if x == mask_x:
                        x = 0
                        y += 1
                        if y == mask_y:
                            return mapping
        raise IndexError("Mask could not be generated properly")


def onepixel(model, args, **kwargs):
    return OnePixel(model, target=args.target, args=args, **kwargs)

def onepixel_latin(model, args, **kwargs):
    return OnePixel(model, init='latinhypercube', target=args.target, args=args, **kwargs)

def onepixel_64(model, args, **kwargs):
    return OnePixel(model, popsize=64, target=args.target, args=args, **kwargs)
