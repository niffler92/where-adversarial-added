import torch
from dataloader import normalize, denormalize

__all__ = ['noise_linf', 'noise_l2']


class Noise:
    def __init__(self, eps=0.031, norm='Linf', args=None, **kwargs):
        self.eps = eps
        self.norm = norm
        self.args = args

    def generate(self, images, labels):
        noise = torch.rand(*images.size())
        if self.norm == 'Linf':
            noise.sign_()
        elif self.norm == 'L2':
            L2_norm = torch.norm(noise.view(noise.size(0), -1), p=2, dim=1)
            noise /= L2_norm.view(-1,1,1,1)
        
        if self.args.cuda:
            noise = noise.cuda()
        if self.args.half:
            noise = noise.half()

        adv_images = denormalize(images, self.args.dataset) + self.eps*denormalize(noise, self.args.dataset)
        adv_images = torch.clamp(adv_images, 0, 1)
        adv_images = normalize(adv_images, self.args.dataset)

        return adv_images


def noise_linf(model, args, **kwargs):
    return Noise(eps=0.031, norm='Linf', args=args, **kwargs)

def noise_l2(model, args, **kwargs):
    return Noise(eps=0.031, norm='L2', args=args, **kwargs)
