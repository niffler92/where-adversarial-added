import torch
import torch.nn as nn
from torch.autograd import Variable
from dataloader import normalize, denormalize

__all__ = ['fgm', 'fgsm', 'ifgsm', 'ifgsm_all', 'bim', 'ilcm', 'pgd', 'pgd_carlini']


class FGM:
    def __init__(self, model, max_iter=1, max_clip=0.01, alpha=None, norm='Linf', target=None,
                 args=None, **kwargs):
        self.model = model
        self.max_iter = max_iter
        self.max_clip = max_clip
        self.alpha = max_clip / max_iter if alpha is None else alpha
        self.norm = norm
        self.target = target
        self.args = args
        if args.domain_restrict:
            self.mask = Variable(kwargs.get('artifact'))
        else:
            self.mask = 1

        # TODO : allow other criterions for computing loss
        self.criterion = nn.CrossEntropyLoss()
        if self.args.half: self.criterion.half()

    def generate(self, images, labels):
        adv_grad = Variable(images.clone(), requires_grad = True)
        adv_nograd = Variable(images.clone())
        adv_images = adv_grad*self.mask + adv_nograd*(1-self.mask)

        if self.target is not None:
            if self.target == -1:
                # set target as the Least-Likely Class
                _, labels = torch.min(self.model(Variable(images)).data, dim=1)
            else:
                # set target as a given integer
                labels = self.target * torch.ones_like(labels)

        for i in range(self.max_iter):
            adv_out = self.model(adv_images)
            loss = self.criterion(adv_out, Variable(labels))
            if self.target is not None:
                loss = -loss

            self.model.zero_grad()
            if adv_grad.grad is not None:
                adv_grad.grad.data.zero_()
            loss.backward()

            if self.norm == 'Linf':
                adv_grad.grad.sign_()
            elif self.norm == 'L1':
                L1_norm = torch.norm(adv_grad.grad.view(labels.size(0), -1), p=1, dim=1)
                adv_grad.grad = adv_grad.grad / L1_norm.view(-1,1,1,1)
            elif self.norm == 'L2':
                L2_norm = torch.norm(adv_grad.grad.view(labels.size(0), -1), p=2, dim=1)
                adv_grad.grad = adv_grad.grad / L2_norm.view(-1,1,1,1)

            adv_images = adv_images + self.alpha * adv_grad.grad
            diff = torch.clamp(denormalize(adv_images, self.args.dataset) - denormalize(Variable(images), self.args.dataset), -self.max_clip, self.max_clip)
            adv_images = torch.clamp(denormalize(images, self.args.dataset) + diff.data, 0, 1)
            adv_images = normalize(adv_images, self.args.dataset)

            adv_grad = Variable(adv_images.clone(), requires_grad = True)
            adv_nograd = Variable(adv_images.clone())
            adv_images = adv_grad*self.mask + adv_nograd*(1-self.mask)

        return adv_images.data, labels


def fgm(model, args, **kwargs):
    return FGM(model, max_iter=1, norm='L2', target=args.target, args=args, **kwargs)

def fgsm(model, args, **kwargs):
    return FGM(model, max_iter=1, max_clip=0.031, alpha=1000, target=args.target, args=args, **kwargs)

def ifgsm(model, args, **kwargs):
    return FGM(model, max_iter=100, target=args.target, args=args, **kwargs)

def ifgsm_all(model, args, **kwargs):
    return [FGM(model, max_iter=100, target=i, args=args, **kwargs) for i in range(args.num_classes)]

def bim(model, args, **kwargs):
    return FGM(model, max_iter=100, alpha=1, max_clip=0.01, target=args.target, args=args, **kwargs)

def pgd(model, args, **kwargs):
    return FGM(model, max_iter=1000, alpha=0.1, max_clip=0.031, norm='Linf', target=args.target, args=args, **kwargs)

def pgd_carlini(model, args, **kwargs):
    return FGM(model, max_iter=1000, alpha=0.1, max_clip=0.031, norm='L2', target=args.target, args=args, **kwargs)

def ilcm(model, args, **kwargs):
    return FGM(model, max_iter=100, alpha=1, target='LL', args=args, **kwargs)
