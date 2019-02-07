import torch
import torch.nn as nn
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable

import numpy as np

from common.summary import AverageMeter
from common.torch_utils import to_np

__all__ = ['deepfool_l1', 'deepfool_l2', 'deepfool_linf']


class DeepFool:
    def __init__(self, model, q=2, max_iter=50, target=None, args=None, **kwargs):
        """
        Args:
            q (int): p-1 / p, l_p norm to use in calculating distance
            max_iter (int): Number of iterations in perturbation
            target: None. This is non-targeted attack.
        """
        self.model = model
        self.max_iter = max_iter
        self.args = args
        self.target = target
        self.num_classes = args.num_classes
        self.q = q
        self.step_meter = AverageMeter()
        if args.domain_restrict:
            self.mask = Variable(kwargs.get('artifact'))
        else:
            self.mask = 1

        self.eps = 10e-5
        self.model.eval()

    def generate(self, images, labels):
        self.original_shape = images[0].shape

        adv_imgs = [self.generate_sample(image, label) for (image, label)
                    in zip(images, labels)]
        adv_imgs = torch.cat(adv_imgs, dim=0)

        return adv_imgs, labels

    def generate_sample(self, image, label):
        """Fooling if label changes. Doesn't matter if source_label=true label or not
        """
        r_sum = 0
        niter = 0
        adv_img = image.clone()
        adv_img = Variable(adv_img.unsqueeze(0))

        output = self.model(adv_img)
        source_label = torch.max(output, 1)[1].data[0]

        adv_label = source_label
        while adv_label == source_label and niter < self.max_iter:
            adv_grad = Variable(adv_img.data, requires_grad=True)
            adv_nograd = Variable(adv_img.data)
            adv_img = adv_grad*self.mask + adv_nograd*(1-self.mask)

            output = self.model(adv_img)
            source_grad = self.get_label_grad(adv_grad, output, source_label)

            dist_min = 1e30
            for k in range(self.num_classes):
                if k != source_label:
                    #zero_gradients(adv_img)  # FIXME: Makes source_grad = 0.. dangerous
                    # but source_grad = 0 works better;;; '-' '^'
                    adv_grad = Variable(adv_img.data, requires_grad=True)
                    adv_nograd = Variable(adv_img.data)
                    adv_img = adv_grad*self.mask + adv_nograd*(1-self.mask)

                    adv_output = self.model(adv_img)
                    target_grad = self.get_label_grad(adv_grad, adv_output, k)

                    w_k = target_grad - source_grad
                    f_k = adv_output[:, k].data - adv_output[:, source_label].data

                    dist = torch.abs(f_k) / w_k.norm(p=self.q)

                    if (dist < dist_min).all():  # torch bool value
                        dist_min = dist
                        w_min = w_k
                        f_min = f_k
                        l = k

            r_i = f_min.abs() / (w_min.norm(p=self.q) ** self.q) * w_min.abs().pow(self.q-1) * w_min.sign()
            adv_img = Variable(adv_img.data.clone() + r_i)
            r_sum += r_i
            niter += 1

            adv_label = torch.max(self.model(adv_img), 1)[1].data[0]
            zero_gradients(adv_grad)

        self.step_meter.update(niter)
        #print("Average step per iter: {}".format(self.step_meter.avg))
        #if adv_label != source_label:
        #    print("Attack Success!")
        #else:
        #    print("Attack Failure")
        return image + r_sum# (1, 3, 32, 32)

    def get_label_grad(self, image, output, label):
        grad_mask = torch.zeros_like(output.data)
        grad_mask[:, label] = 1

        output.backward(grad_mask, retain_graph=True)
        output.detach()

        del output

        return image.grad.data


def deepfool_l2(model, args, **kwargs):
    p = 2
    return DeepFool(model, q=p/(p-1), max_iter=args.max_iter, args=args, **kwargs)


def deepfool_l1(model, args, **kwargs):
    """
        Currently has a bug in pytorch. Calculates infinity norm as 1
        https://github.com/pytorch/pytorch/issues/6817
    """
    return DeepFool(model, q=np.inf, max_iter=args.max_iter, args=args, **kwargs)


def deepfool_linf(model, args, **kwargs):
    return DeepFool(model, q=1, max_iter=args.max_iter, args=args, **kwargs)
