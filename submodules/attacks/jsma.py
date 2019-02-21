import torch
import numpy as np

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

        self.increase = bool(theta>0)
        self.jacobians = None

        if self.target is None:
            self.target = -1

    def generate(self, images, labels):
        self.model.eval()
        self.n_features = int(np.prod(images.shape[1:]))
        self.original_shape = images[0].shape

        adv_imgs = [self.generate_sample(image, label) for i, (image, label)
                    in enumerate(zip(images, labels))]
        adv_imgs = torch.stack(adv_imgs)

        return adv_imgs

    def generate_sample(self, image, label):
        self.clip_max = torch.max(image) if self.clip_max is None else self.clip_max
        self.clip_min = torch.min(image) if self.clip_min is None else self.clip_min

        adv_img = image.clone()
        adv_img = adv_img.view(1, self.n_features)

        search_domain = self.get_search_domain(adv_img)
        cur_iter = 0
        max_iter = np.floor(self.n_features * self.gamma / 2)  # 2 pixels per iter

        adv_input = self.to_org_shape(adv_img)
        adv_output = self.model(adv_input).detach()
        source = torch.topk(adv_output, k=1)[1][0]

        if self.target == -1:  # Least likely label
            target = int(torch.min(adv_output, dim=1)[1][0])
        else:
            target = self.target

        while (source != target).all() and cur_iter < max_iter and search_domain:
            jacobian = self.get_jacobian(self.model, adv_img)
            grads_target = jacobian[target]  # (dF_target(x) / d_x)
            grads_others = torch.sum(jacobian, dim=0) - grads_target

            p1, p2, search_domain = self.saliency_map(grads_target, grads_others, search_domain)
            adv_img = self.perturb(p1, p2, adv_img)

            if (adv_img[0, p1] == self.clip_min or adv_img[0, p1] == self.clip_max): search_domain.discard(p1)
            if (adv_img[0, p2] == self.clip_min or adv_img[0, p2] == self.clip_max): search_domain.discard(p2)

            adv_input = self.to_org_shape(adv_img)
            source = torch.topk(self.model(adv_input).detach(), k=1)[1]
            cur_iter += 1

        torch.cuda.empty_cache()

        return adv_img.view(*self.original_shape)

    def get_jacobian(self, model, image):
        image = self.to_org_shape(image)
        output = model(image)  # (B=1, num_classes), different varname bcz of leaf variable grad
        output = output.float()

        jacobian = torch.zeros(self.args.num_classes, 1, self.n_features)
        grad_output = torch.zeros_like(output)

        for i in range(self.args.num_classes):
            grad_output.zero_()
            grad_output[:, i] = 1
            image_tmp = image.clone()
            image_tmp.requires_grad_()

            output = model(image_tmp)  # (B=1, num_classes), different varname bcz of leaf variable grad
            output.backward(grad_output)
            jacobian[i] = image_tmp.grad.view(1,-1)

            del output, image_tmp
            torch.cuda.empty_cache()

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

        domain_idx = list(range(self.n_features))
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


def jsma(model, args, **kwargs):
    return JSMA(model, args=args, **kwargs)
