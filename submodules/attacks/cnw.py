import torch

from common.torch_utils import get_optimizer
from dataloader import normalize, denormalize

__all__ = ['cwl0', 'cwl2', 'cwli']


class CnW:
    def __init__(self, model, inner_iter=100, outer_iter=10, c0 = 1e-3,
                 max_clip=1, min_clip=0, tau0=1.0, kappa=0, norm='L2',
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

        self.c_rate = c_rate
        self.tau_rate = tau_rate
        self.max_eps = max_eps

        self.args = args

    def generate(self, images, labels):
        self.model.eval()
        images = denormalize(images, self.args.dataset)
        _, labels = torch.max(labels, dim=1)
        
        outer_adv_images = images.clone()
        outer_Lp = torch.ones(images.size(0)) * 1e10
        if self.args.cuda: 
            outer_Lp = outer_Lp.cuda()

        self.lower = torch.zeros(self.args.batch_size)
        self.upper = torch.ones(self.args.batch_size) * 1e10
        if self.args.cuda:
            self.lower = self.lower.cuda()
            self.upper = self.upper.cuda()

        c = torch.ones(self.args.batch_size)*self.c0
        tau = torch.ones(self.args.batch_size)*self.tau0
        if self.args.cuda:
            c = c.cuda()
            tau = tau.cuda()

        # perform binary search for the best c, i.e. constant for confidence loss
        for binary_step in range(self.outer_iter):
            update = torch.zeros(images.size(0))
            valid = torch.ones(images.size(0), 1, images.size(2), images.size(3))
            if self.args.cuda: 
                update = update.cuda()
                valid = valid.cuda()

            # variables used only inside the binary search loop
            inner_adv_latent = self.unclip(images)
            inner_adv_latent.requires_grad_()

            inner_adv_images = self.clip(inner_adv_latent)
            inner_adv_out = self.model(normalize(inner_adv_images, self.args.dataset))
            inner_Lp = torch.ones(images.size(0))*1e10
            inner_grad = torch.zeros_like(images)
            if self.args.cuda: 
                inner_Lp = inner_Lp.cuda()

            optimizer = get_optimizer(self.args.optimizer, [inner_adv_latent], self.args)

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
                Z_diff = Z_t - Z_nt
                conf_loss = torch.max(Z_diff, torch.ones_like(Z_diff) * (-self.kappa))

                loss = Lp_loss + torch.dot(c, conf_loss)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                grad = inner_adv_latent.grad
                inner_adv_images = self.clip(inner_adv_latent)*valid + images*(1-valid)
                inner_adv_out = self.model(normalize(inner_adv_images, self.args.dataset))
                success = (torch.max(inner_adv_out, dim=1)[1] != labels)

                inner_update = ((inner_Lp > Lp)*success).float()
                outer_update = ((outer_Lp > Lp)*success).float()
                update = update + inner_update

                inner_Lp += inner_update*(Lp - inner_Lp)
                outer_Lp += outer_update*(Lp - outer_Lp)

                inner_update = inner_update.view(-1,1,1,1)
                inner_grad += inner_update*(grad - inner_grad)

                outer_update = outer_update.view(-1,1,1,1)
                outer_adv_images = outer_update*inner_adv_images + \
                                   (1 - outer_update)*outer_adv_images

            c = self.binary_search(c, update)
            abs_diff = torch.abs(inner_adv_images - images)
            if self.norm == 'L0':
                totalchange = torch.sum(abs_diff*torch.abs(inner_grad), dim=1)
                valid = (totalchange > self.max_eps)
                valid = valid.view((images.size(0), 1, images.size(2), images.size(3)))
            elif self.norm == 'Li':
                actual_tau, _ = torch.max(abs_diff.view(images.size(0),-1), dim=1)
                tau = self.reduce_tau(tau, actual_tau, update)

        adv_images = normalize(outer_adv_images, self.args.dataset).detach()
        return adv_images

    def clip(self, images):
        images = torch.tanh(images)
        images = images * (self.max_clip - self.min_clip)/2 + (self.max_clip + self.min_clip)/2
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
        c[~init] = ((self.upper + self.lower)/2)[~init]
        return c

    def reduce_tau(self, tau, actual_tau, update):
        update = update.float()
        tau = torch.min(tau, actual_tau)*self.tau_rate*update + tau*(1-update)
        return tau


def cwl0(model, args, **kwargs):
    args.learning_rate = 1e-2
    args.optimizer = 'Adam'
    return CnW(model, norm='L0', c0=1e-3, c_rate=1, args=args, **kwargs)

def cwl2(model, args, **kwargs):
    args.learning_rate = 1e-2
    args.optimizer = 'Adam'
    return CnW(model, norm='L2', c0=1e-3, c_rate=10, args=args, **kwargs)

def cwli(model, args, **kwargs):
    args.learning_rate = 5e-3
    args.optimizer = 'Adam'
    return CnW(model, norm='Li', c0=1e-5, c_rate=1, args=args, **kwargs)
