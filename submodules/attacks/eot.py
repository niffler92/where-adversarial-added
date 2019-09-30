import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from dataloader import normalize, denormalize
from common.torch_utils import get_optimizer, to_var
from submodules import attacks, defenses

__all__ = ['eot']


class EOT:
    def __init__(self, model, attack='pgd_carlini', defense='pixelshift', nsamples=100, args=None, **kwargs):
        """
        Expectation Over Transformation
        Args:
            attack (str): type of attack to ensemble over
            defense (str): type of defense to conduct expectation on
            nsamples (int): Number of samples to ensemble
        """
        assert attack in dir(attacks)
        assert defense in dir(defenses)
        self.model = model
        self.attack = getattr(attacks, attack)(model, args, **kwargs)
        self.defense = getattr(defenses, 'pixelshift')(model, args, **kwargs)  # FIXME
        self.nsamples = nsamples
        self.args = args

        self.cuda = self.args.cuda

    def generate(self, images, labels):
        adv_imgs = [self.generate_sample(image, label) for image, label in zip(images, labels)]
        adv_imgs = torch.stack(adv_imgs, dim=0)

        return adv_imgs, labels

    def generate_sample(self, image, label):
        # PGD Obfuscated Gradients setting
        alpha = 0.1
        max_clip = 0.031
        adv_img = image.clone()
        criterion = nn.CrossEntropyLoss()
        label = torch.LongTensor([label])

        if isinstance(adv_img, Variable):
            adv_img = adv_img.data
        adv_img = Variable(adv_img, requires_grad=True)  # Start of graph

        for i in range(self.args.eot_iter):

            ensemble_images = torch.cat([self.defense.generate(adv_img.unsqueeze(0), label)[0] for i in range(self.nsamples)], dim=0)
            ensemble_labels = to_var(label.repeat(self.nsamples), self.cuda)

            ensemble_outputs = self.model(ensemble_images)
            ensemble_loss = criterion(ensemble_outputs, ensemble_labels)
            if adv_img.grad is not None:
                adv_img.grad.data.zero_()
            ensemble_loss.backward()

            if self.args.eot_norm == 'linf':
                adv_img.grad.sign()
            elif self.args.eot_norm == 'l2':
                L2_norm = torch.norm(adv_img.grad.view(label.size(0), -1), p=2, dim=1)
                adv_img.grad = adv_img.grad / L2_norm.view(-1,1,1)
            else:
                raise ValueError

            adv_img = adv_img + alpha * adv_img.grad
            diff = torch.clamp(denormalize(adv_img, self.args.dataset) - denormalize(Variable(image), self.args.dataset), -max_clip, max_clip)
            adv_img = torch.clamp(denormalize(image, self.args.dataset) + diff.data, 0, 1)
            adv_img = Variable(normalize(adv_img, self.args.dataset)[0], requires_grad=True)

        return adv_img


def eot(model, args, **kwargs):
    return EOT(model, attack=args.eot_attack, defense=args.defense, nsamples=args.nsamples, args=args, **kwargs)
