from pathlib import Path
from datetime import datetime
import os
import numpy as np
from itertools import islice
import pickle
from urllib.request import urlopen
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
cmap = matplotlib.rcParams['image.cmap']
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from PIL import Image

from settings import PROJECT_ROOT
from common.logger import Logger
from common.torch_utils import to_np, to_var, get_model
from common.attack_utils import get_artifact
import submodules.attacks as attacks
import submodules.defenses as defenses
from dataloader import denormalize


class Analysis:
    """Plot contour maps for the decision boundaries of model"""
    def __init__(self, val_loader, args, **kwargs):
        self.loader = islice(iter(val_loader), args.img_first, args.img_last+1)
        self.model = get_model(args)
        self.art = to_np(get_artifact(self.model, val_loader, args) > 0)
        #self.attack = getattr(attacks, args.attack)(self.model, args)
        #self.defense = getattr(defenses, args.defense)(self.model, args)
        self.eps = np.linspace(-args.max_eps, args.max_eps, args.len_eps)
        self.args = args
        self.kwargs = kwargs

        self.log_path = (
                PROJECT_ROOT / Path("experiments") /
                Path(datetime.now().strftime("%Y%m%d%H%M%S") + "-")
                )
        if not Path.exists(self.log_path):
            Path(self.log_path).mkdir(parents=True, exist_ok=True)
        self.logger = Logger("analysis", self.log_path, args.verbose)
        self.logger.log("Checkpoint files will be saved in {}".format(self.log_path))

        self.logger.add_level('ANALYSIS', 21)

    def analysis(self):
        names = self.get_names()
        #levels = np.arange(-1, self.args.num_classes)
        figsize = (5,5)
        criterion = nn.CrossEntropyLoss()

        n_img = self.args.img_last - self.args.img_first + 1
        preds = np.zeros((len(self.eps), len(self.eps)))
        self.model.eval()
        for k in range(n_img):
            (image, label) = next(self.loader)
            if self.args.cuda:
                image = image.cuda()
                label = label.cuda()
            #adv_image = self.attack.generate(image, label)[0]
            #adv_diff = to_np(adv_image - image)
            #def_image = self.defense.generate(adv_image, label)[0]
            #def_diff = to_np(def_image - image)

            image = to_var(image)
            label = to_var(label)
            image.requires_grad = True
            output = self.model(image)
            loss = criterion(output, label)
            loss.backward()

            grad = to_np(image.grad)
            art_image = grad*self.art
            art_image = art_image/np.linalg.norm(art_image)
            nart_image = grad*(1 - self.art)
            nart_image = nart_image/np.linalg.norm(nart_image)

            #X = np.vstack([art_image.flatten(), nart_image.flatten()]).T
            #X = np.linalg.inv(X.T@X)@X.T
            #adv_loc = X@adv_diff.flatten()
            #def_loc = X@def_diff.flatten()

            for i, e1 in enumerate(self.eps):
                for j, e2 in enumerate(self.eps):
                    eps_image = image + to_var(torch.FloatTensor(art_image*e1 + nart_image*e2))
                    _, pred = torch.max(self.model(eps_image), 1)
                    preds[i,j] = pred.data[0]

            # save preds data
            self.logger.check_savedir()
            np.save(self.logger.log_dir + '/numpy/preds_' + str(k), preds)

            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            ax.pcolormesh(self.eps, self.eps, preds, cmap=cmap, vmin=0, vmax=self.args.num_classes-1)
            # ax.contourf(self.eps, self.eps, preds, levels=levels, cmap=cmap)
            # ax.contour(self.eps, self.eps, preds, colors='k', linewidths=2, linestyles='solid')
            ax.axvline(0, color='k', linestyle=':', alpha=0.5)
            ax.axhline(0, color='k', linestyle=':', alpha=0.5)

            ax.set_xlim(-self.args.max_eps, self.args.max_eps)
            ax.set_ylim(-self.args.max_eps, self.args.max_eps)

            handles = []
            colors = matplotlib.cm.get_cmap(cmap)
            for label in np.unique(preds):
                color = colors(label/(self.args.num_classes-1))
                handles.append(Patch(facecolor=color, edgecolor=None, label=names[label]))
            ax.legend(handles=handles, loc='upper left')

            """
            handles = [Line2D([0], [0], marker='s', color='k', label='Clean'),
                       Line2D([0], [0], marker='x', color='k', label='Attack'),
                       Line2D([0], [0], marker='o', color='k', label='Defense')]
            ax.legend(handles=handles, loc='upper left')

            axin = zoomed_inset_axes(ax, self.args.max_eps/8, loc='lower right')
            axin.contourf(self.eps, self.eps, preds, levels=[-1,0,1], cmap=cmap)
            axin.axvline(0, color='k', linestyle=':', alpha=0.5)
            axin.axhline(0, color='k', linestyle=':', alpha=0.5)
            axin.plot([0], [0], marker='s', color='k', label='Clean')
            axin.plot([adv_loc[1]], [adv_loc[0]], marker='x', color='k', label='Attack')
            axin.plot([def_loc[1]], [def_loc[0]], marker='o', color='k', label='Defense')

            axin.set_xlim(-1.5,1.5)
            axin.set_ylim(-1.5,1.5)
            axin.xaxis.set_visible(False)
            axin.yaxis.set_visible(False)
            mark_inset(ax, axin, loc1=1, loc2=3, fc="None", ec="0.7")
            """
            fig.tight_layout()

            fig.savefig('contour.png')
            fig = np.asarray(Image.open('contour.png'))
            os.remove('contour.png')

            img_dict = {'Decision Boundary {}'.format(k+self.args.img_first): np.transpose(fig,[2,0,1])}
            img_org_dict = {'Image Origin {}'.format(k+self.args.img_first): denormalize(image.data, self.args.dataset)[0]}
            self.logger.image_summary(img_dict, k+self.args.img_first, save=True)
            self.logger.image_summary(img_org_dict, k+self.args.img_first, save=True)
            self.logger.log("Processed Image No.{}".format(k+self.args.img_first), 'ANALYSIS')


    def get_names(self):
        if self.args.dataset == 'CIFAR10':
            names = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
        elif self.args.dataset == 'ImageNet':
            names = pickle.load(urlopen('https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl') )
            for k, v in names.items():
                if v.find(',') != -1:
                    names[k] = v[:v.find(',')]
        else:
            raise NotImplementedError

        return names
