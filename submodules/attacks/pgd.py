import torch
import torch.nn as nn
from dataloader import normalize, denormalize

__all__ = ['pgd', 'pgd_carlini']


class PGD:
    def __init__(self, model, max_iter=1, max_clip=0.01, alpha=None, norm='Linf',
                 args=None, **kwargs):
        self.model = model
        self.max_iter = max_iter
        self.max_clip = max_clip
        self.alpha = max_clip / max_iter if alpha is None else alpha
        self.norm = norm
        self.args = args

        self.criterion = nn.CrossEntropyLoss()
        if self.args.half: self.criterion.half()

    def generate(self, images, labels):
        adv_images = images.clone()
        for i in range(self.max_iter):
            adv_images.requires_grad_()
            adv_out = self.model(adv_images)
            loss = self.criterion(adv_out, labels)

            self.model.zero_grad()
            loss.backward()

            if self.norm == 'Linf':
                adv_images.grad.sign_()
            elif self.norm == 'L2':
                L2_norm = torch.norm(adv_images.grad.view(labels.size(0), -1), p=2, dim=1)
                adv_images.grad = adv_images.grad / L2_norm.view(-1,1,1,1)

            adv_images = adv_images + self.alpha * adv_images.grad
            diff = torch.clamp(denormalize(adv_images, self.args.dataset) - denormalize(images, self.args.dataset), -self.max_clip, self.max_clip)
            adv_images = torch.clamp(denormalize(images, self.args.dataset) + diff.data, 0, 1)
            adv_images = normalize(adv_images, self.args.dataset)

        return adv_images.detach(), labels


def pgd(model, args, **kwargs):
    return PGD(model, max_iter=1000, alpha=0.1, max_clip=0.031, args=args, **kwargs)

def pgd_carlini(model, args, **kwargs):
    return PGD(model, max_iter=1000, alpha=0.1, max_clip=0.031, norm='L2', args=args, **kwargs)
