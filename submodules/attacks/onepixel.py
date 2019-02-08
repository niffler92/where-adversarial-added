import torch
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from common.differential_evolution import differential_evolution
from dataloader import normalize, denormalize


__all__ = ['onepixel', 'onepixel_latin']


class OnePixel:
    def __init__(self, model, n_pix=1, strategy='rand1bin', max_iter=100, popsize=400,
                 init='normal', args=None, **kwargs):
        """An Attacker algorithm must have a generate method
        """
        self.model = model
        self.n_pix = n_pix
        self.strategy = strategy
        self.max_iter = max_iter
        self.popsize = popsize
        self.init = init
        self.args = args
        self.kwargs = kwargs

    def generate(self, images, labels):
        """Generate adversarial images
        """
        _, preds = torch.max(self.model(images), dim=1)
        images = denormalize(images, self.args.dataset)*255
        bounds = [(0, images[0].size(1)-1), (0, images[0].size(2)-1),
                  (0, 255), (0, 255), (0, 255)] * self.n_pix

        adv_images = []
        adv_labels = []

        for i in range(len(images)):
            self.image = images[i]
            self.label = int(preds[i])

            self.convergence = False

            if self.init == 'normal':
                x_loc = np.random.uniform(0, images[0].size(1), self.n_pix*self.popsize)
                y_loc = np.random.uniform(0, images[0].size(2), self.n_pix*self.popsize)
                val = np.array(np.split(np.random.normal(128, 127, self.n_pix*self.popsize*3),3))
                init = np.array(np.split(np.vstack((x_loc, y_loc, val)), self.n_pix, axis=1))
                init = np.transpose(init.reshape(-1, self.popsize))

            else:
                init = self.init

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

        return torch.stack(adv_images)

    def perturb(self, x):
        adv_images = self.image.clone().unsqueeze(0)
        pixs = np.transpose(x)
        pixs = np.array(np.split(pixs, self.n_pix))

        if pixs.ndim == 2:
            pixs = pixs.reshape(*pixs.shape,1)
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
                img[:,x,y] = val[:,i]

        adv_images = normalize(adv_images/255, self.args.dataset)

        return adv_images

    def optimize(self, x):
        adv_images = self.perturb(x)
        out = self.model(adv_images)
        out = F.softmax(out, dim=1)

        _, labels = torch.max(out, dim=1)
        if any(labels != torch.ones_like(labels)*self.label):
            self.convergence = True
        out = out[:,self.label]
        out = out.detach().cpu().numpy()

        return out

    def callback(self, x, convergence):
        return self.convergence

    
def onepixel(model, args, **kwargs):
    return OnePixel(model, args=args, **kwargs)

def onepixel_latin(model, args, **kwargs):
    return OnePixel(model, init='latinhypercube', args=args, **kwargs)
