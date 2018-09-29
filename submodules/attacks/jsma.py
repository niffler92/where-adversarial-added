import torch
from torch.autograd import Variable
import numpy as np
from torch.autograd.gradcheck import zero_gradients

from common.torch_utils import to_np, to_var
from common.attack_utils import get_cb_grid
from common.summary import AverageMeter

__all__ = ['jsma']


class JSMA:
    def __init__(self, model, target=None, theta=1, gamma=0.1, clip_min=None, clip_max=None,
                 args=None, **kwargs):
        """Jacobian Saliency Map based Attack
            https://arxiv.org/pdf/1511.07528.pdf

        Args:
            target: If target is None, non-targeted attacks will be done with least likely label
            theta: change made to pixels
            gamma: maximum distortion
            clip_min/max: Value to clip for perturbed image
            top_overlap (int): Initialization on checkerboard, default None
        """
        self.model = model
        self.target = target
        self.theta = theta
        self.gamma = gamma
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.args = args
        self.kwargs = kwargs
        self.cuda = self.args.cuda

        self.increase = bool(theta>0)
        self.jacobians = None
        self.step_meter = AverageMeter()

        self.model.eval()
        for p in self.model.parameters():  # FIXME
            p.requires_grad = False

        if self.target is None:
            self.target = -1

        print("JSMA attack with target: {}".format(self.target))

    def generate(self, images, labels):
        self.n_features = int(np.prod(images.shape[1:]))
        self.original_shape = images[0].shape

        adv_imgs = [self.generate_sample(image, label) for i, (image, label)
                    in enumerate(zip(images, labels))]
        adv_imgs = torch.stack(adv_imgs)

        if self.target == -1:  # Least likely label
            _, labels = torch.min(self.model(self.to_var(images, self.cuda, True)).data, dim=1)
            return adv_imgs, labels
        return adv_imgs, self.target * torch.ones_like(labels)

    def generate_sample(self, image, label):
        self.clip_max = torch.max(image) if self.clip_max is None else self.clip_max
        self.clip_min = torch.min(image) if self.clip_min is None else self.clip_min

        adv_img = image.clone()
        adv_img = adv_img.view(1, self.n_features)

        search_domain = self.get_search_domain(adv_img)
        cur_iter = 0
        max_iter = np.floor(self.n_features * self.gamma / 2)  # 2 pixels per iter

        adv_input = self.to_var(self.to_org_shape(adv_img), self.cuda, True)
        adv_output = self.model(adv_input).detach()
        source = torch.topk(adv_output, k=1)[1].data[0]

        # prob, source = torch.topk(self.model(adv_input).detach(), k=1)
        # source = source.data[0]

        if self.target == -1:  # Least likely label
            target = int(torch.min(adv_output, dim=1)[1].data[0])
        else:
            target = self.target

        del adv_input

        while (source != target).all() and cur_iter < max_iter and search_domain:
            jacobian = self.get_jacobian(self.model, adv_img)
            grads_target = jacobian[target]  # (dF_target(x) / d_x)
            grads_others = torch.sum(jacobian, dim=0) - grads_target

            p1, p2, search_domain = self.saliency_map(grads_target, grads_others, search_domain)
            adv_img = self.perturb(p1, p2, adv_img)

            if (adv_img[0, p1] == self.clip_min or adv_img[0, p1] == self.clip_max): search_domain.discard(p1)
            if (adv_img[0, p2] == self.clip_min or adv_img[0, p2] == self.clip_max): search_domain.discard(p2)

            adv_input = self.to_var(self.to_org_shape(adv_img), self.cuda, True)
            source = torch.topk(self.model(adv_input).detach(), k=1)[1]
            del adv_input
            cur_iter += 1

            if cur_iter % 20 == 0 or cur_iter == 1:
                print("[{}/{}] - Target: {}, source: {}".format(cur_iter, max_iter, target, to_np(source)[0]))
                print("True: ", label)

        self.step_meter.update(cur_iter)
        print("Average step per iter: {}".format(self.step_meter.avg))
        if (source == target).all():
            print("Success!")  # , source: {}, target: {}, true: {}".format(source[0], target, label))
        else:
            print("Failed!")

        torch.cuda.empty_cache()

        return adv_img.view(*self.original_shape)

    def get_jacobian(self, model, image):
        image = self.to_org_shape(image)
        image = Variable(image, requires_grad=True)
        image_cuda = image.cuda() if self.args.cuda else image
        output = model(image_cuda)  # (B=1, num_classes), different varname bcz of leaf variable grad
        output = output.float()

        jacobian = torch.zeros(self.args.num_classes, 1, self.n_features)
        grad_output = torch.zeros_like(output.data)

        #if self.args.cuda:
        #    jacobian = jacobian.cuda()
        #    grad_output = grad_output.cuda()

        for i in range(self.args.num_classes):
            #zero_gradients(image)
            grad_output.zero_()
            grad_output[:, i] = 1
            image = Variable(image.data, requires_grad=True)
            image_cuda = image.cuda() if self.args.cuda else image

            output = model(image_cuda)  # (B=1, num_classes), different varname bcz of leaf variable grad
            output.backward(grad_output, retain_graph=True)
            jacobian[i] = image.grad.data

            del output
            torch.cuda.empty_cache()

        del image
        return torch.transpose(jacobian, dim0=0, dim1=1).squeeze(0)  # (n_class, n_features)

    def saliency_map(self, grads_target, grads_others, search_domain):
        """
        Args
            grads_target: dF_t(X) / dx_i  (n_features) (t = target)
            grads_others: \sum_(j!=t) dF_j(X) / dx_i (n_features)
        """

        # Remove already used indexes from selection
        invalid = list(set(range(self.n_features)) - search_domain)
        increase_coef = (2 * int(self.increase) - 1)
        if invalid:
            grads_target[invalid] = - increase_coef * torch.max(torch.abs(grads_target))
            grads_others[invalid] = increase_coef * torch.max(torch.abs(grads_others))

        target_sum = grads_target.view(1, -1) + grads_target.view(-1, 1)  # \alpha
        others_sum = grads_others.view(1, -1) + grads_others.view(-1, 1)  # \beta

        if self.increase:
            mask = (target_sum > 0) * (others_sum < 0)
        else:
            mask = (target_sum < 0) * (others_sum > 0)

        scores = (-target_sum * others_sum) * mask.float()
        diag_mask = 1 - torch.eye(self.n_features)
        #if self.args.cuda: diag_mask = diag_mask.cuda()
        scores = scores * diag_mask  # Exclude diagonal (only 1 feature)

        max_idx = np.argmax(scores)
        p1, p2 = max_idx % self.n_features, max_idx // self.n_features

        return p1, p2, search_domain

    def get_search_domain(self, adv_img):
        """
        Args:
            top_overlap (int): Top kth overlap checkerboard to use as search domain
        """
        width = self.original_shape[1]
        height = self.original_shape[2]
        n_change = int(np.floor(self.n_features * self.gamma))

        if self.args.domain_restrict:
            mask = to_np(self.kwargs.get('artifact'))
            mask = np.tile(mask,(3,1,1)).reshape([1, -1])
            domain_idx = np.argwhere(mask.ravel() == 1).ravel().tolist()
            #overlap1 = sorted(np.unique(cb_grid), reverse=True)[0]
            #overlap2 = sorted(np.unique(cb_grid), reverse=True)[1]
            #domain_idx = np.argwhere(np.logical_or(cb_grid.ravel() == overlap1, cb_grid.ravel() == overlap2)).ravel().tolist()
        else:
            domain_idx = list(range(self.n_features))

        # Domain Resctriction
        #if self.args.domain_restrict:
        #    domain_idx = np.random.choice(domain_idx, n_change, replace=False)

        if self.increase:
            return set(i for i in domain_idx if adv_img[0, i] < self.clip_max)
        else:
            return set(i for i in domain_idx if adv_img[0, i] > self.clip_min)

    def perturb(self, i, j, img):
        if self.increase:
            img[0, i] = min(self.clip_max, img[0, i] + self.theta)
            img[0, j] = min(self.clip_max, img[0, j] + self.theta)
        else:
            img[0, i] = max(self.clip_min, img[0, i] - self.theta)
            img[0, j] = max(self.clip_min, img[0, j] - self.theta)

        return img

    def to_org_shape(self, img):
        return img.view(*self.original_shape).unsqueeze(0)

    def to_var(self, x, cuda, volatile=False):
        """For CPU inference manual cuda setting is needed
        """
        if cuda:
            x = x.cuda()
        return torch.autograd.Variable(x, volatile=volatile)


def jsma(model, args, **kwargs):
    return JSMA(model, target=args.target, gamma=args.gamma, args=args, **kwargs)
