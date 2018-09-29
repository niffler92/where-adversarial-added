import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from common.torch_utils import get_optimizer, to_var
from dataloader import data_stats, normalize, denormalize

__all__ = ['cwl0', 'cwl2', 'cwli']


class CnW:
    def __init__(self, model, inner_iter=100, outer_iter=10, c0 = 1e-3,
                 max_clip=1, min_clip=0, tau0=1.0, kappa=0, norm='L2', target=None,
                 c_rate = 10, tau_rate = 0.9, max_eps=0.01, args=None, **kwargs):
        self.model = model
        self.inner_iter = inner_iter
        self.outer_iter = outer_iter
        self.c0 = c0
        self.max_clip = max_clip
        self.min_clip = min_clip
        self.tau0 = tau0
        self.kappa = kappa
        self.norm = norm
        self.target = target
        self.c_rate = c_rate
        self.tau_rate = tau_rate
        self.max_eps = max_eps
        self.args = args
        if args.domain_restrict:
            self.mask = to_var(kwargs.get('artifact'))
        else:
            self.mask = 1

    def generate(self, images, labels):
        if self.target is not None:
            if self.target == -1:  # Least likely method
                _, labels = torch.min(self.model(to_var(images)).data, dim=1)
            else:
                labels = self.target * torch.ones_like(labels)
        labels = to_var(labels)
        # images = to_var(images*self.std.data + self.mean.data)
        images = denormalize(to_var(images), self.args.dataset)
        outer_adv_images = images.data.clone()
        outer_Lp = torch.ones(images.size(0)) * 1e10
        if self.args.cuda: outer_Lp = outer_Lp.cuda()

        self.lower = torch.zeros(self.args.batch_size)
        self.upper = torch.ones(self.args.batch_size) * 1e10
        if self.args.cuda:
            self.lower = self.lower.cuda()
            self.upper = self.upper.cuda()
        c = to_var(torch.ones(self.args.batch_size)*self.c0)
        tau = to_var(torch.ones(self.args.batch_size)*self.tau0)

        # perform binary search for the best c, i.e. constant for confidence loss
        for binary_step in range(self.outer_iter):

            update = torch.zeros(images.size(0))
            if self.args.cuda: update = update.cuda()
            valid = to_var(torch.ones(images.size(0), 1, images.size(2), images.size(3)))

            # variables used only inside the binary search loop
            inner_adv_grad = to_var(self.unclip(images.data))
            inner_adv_grad.requires_grad = True
            inner_adv_nograd = to_var(self.unclip(images.data))
            inner_adv_latent = inner_adv_grad*self.mask + inner_adv_nograd*(1-self.mask)

            inner_adv_images = self.clip(inner_adv_latent)
            inner_adv_out = self.model(normalize(inner_adv_images, self.args.dataset))
            inner_Lp = torch.ones(images.size(0))*1e10
            inner_grad = torch.zeros_like(images.data)
            if self.args.cuda: inner_Lp = inner_Lp.cuda()

            optimizer = get_optimizer(self.args.optimizer, [inner_adv_grad], self.args)

            for step in range(self.inner_iter):
                diff = (inner_adv_images - images).view(images.size(0),-1)
                if self.norm == 'Li':
                    Lp = torch.max(torch.abs(diff), tau.view(-1,1))
                    Lp = torch.sum(Lp, dim=1)
                else:
                    Lp = torch.norm(diff, p=2, dim=1)**2
                Lp_loss = torch.sum(Lp)

                Z_t = inner_adv_out.gather(1, labels.view(-1,1)).squeeze(1)
                Z_nt, _ = torch.max(inner_adv_out.scatter(1, labels.view(-1,1), -1e10), dim=1)
                Z_diff = Z_nt - Z_t
                if self.target is None:
                    Z_diff = -Z_diff
                conf_loss = torch.max(Z_diff, torch.ones_like(Z_diff) * (-self.kappa))

                loss = Lp_loss + torch.dot(c, conf_loss)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                grad = inner_adv_grad.grad
                inner_adv_latent = inner_adv_grad*self.mask + inner_adv_nograd*(1-self.mask)
                inner_adv_images = self.clip(inner_adv_latent) * valid + images * (1-valid)
                # inner_adv_out = self.model((inner_adv_images - self.mean)/self.std)
                inner_adv_out = self.model(normalize(inner_adv_images, self.args.dataset))
                success = (torch.max(inner_adv_out, dim=1)[1] == labels).data
                if self.target is None:
                    success = ~success
                inner_update = ((inner_Lp > Lp.data) * success).float()
                outer_update = ((outer_Lp > Lp.data) * success).float()
                update = update + inner_update

                inner_Lp += inner_update * (Lp.data - inner_Lp)
                outer_Lp += outer_update * (Lp.data - outer_Lp)

                inner_update = inner_update.view(-1,1,1,1)
                inner_grad += inner_update * (grad.data - inner_grad)

                outer_update = outer_update.view(-1,1,1,1)
                outer_adv_images = outer_update * inner_adv_images.data + \
                                   (1 - outer_update) * outer_adv_images

            c = self.binary_search(c, update)
            abs_diff = torch.abs(inner_adv_images - images)
            if self.norm == 'L0':
                totalchange = torch.sum(abs_diff.data * torch.abs(inner_grad), dim=1)
                valid = (totalchange > self.max_eps)
                valid = valid.view((images.size(0), 1, images.size(2), images.size(3)))
            elif self.norm == 'Li':
                actual_tau, _ = torch.max(abs_diff.view(images.size(0),-1), dim=1)
                tau = self.reduce_tau(tau, actual_tau, update)

        # adv_images = (outer_adv_images - self.mean.data) / self.std.data
        adv_images = normalize(to_var(outer_adv_images), self.args.dataset)
        return adv_images.data, labels

    def clip(self, images):
        images = torch.tanh(images)
        images = images * (self.max_clip - self.min_clip)/2 + (self.max_clip + self.min_clip)/2
        if self.args.cuda: images = images.cuda()
        return images

    def unclip(self, images):
        images = (images - (self.max_clip + self.min_clip)/2)/(self.max_clip - self.min_clip)*2
        images = torch.clamp(images, min=-0.999, max=0.999)
        images = torch.log((1 + images)/(1 - images))/2
        return images

    def binary_search(self, c, update):
        update = update.byte()
        self.upper[update] = torch.min(self.upper, c.data)[update]
        self.lower[~update] = torch.max(self.lower, c.data)[~update]
        init = ~update * (self.upper == 1e10)
        c[init] = c[init] * self.c_rate
        c[~init] = to_var((self.upper + self.lower)/2)[~init]
        return c

    def reduce_tau(self, tau, actual_tau, update):
        update = to_var(update.float())
        tau = torch.min(tau, actual_tau)*self.tau_rate*update + tau*(1-update)
        return tau


def cwl0(model, args, **kwargs):
    args.learning_rate = 1e-2
    args.optimizer = 'Adam'
    return CnW(model, norm='L0', c0=1e-3, c_rate=1, target=args.target, args=args, **kwargs)

def cwl2(model, args, **kwargs):
    args.learning_rate = 1e-2
    args.optimizer = 'Adam'
    return CnW(model, norm='L2', c0=1e-3, c_rate=10, target=args.target, args=args, **kwargs)

def cwli(model, args, **kwargs):
    args.learning_rate = 5e-3
    args.optimizer = 'Adam'
    return CnW(model, norm='Li', c0=1e-5, c_rate=1, target=args.target, args=args, **kwargs)
