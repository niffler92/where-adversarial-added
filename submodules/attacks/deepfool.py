import torch
import numpy as np

__all__ = ['deepfool_l1', 'deepfool_l2', 'deepfool_linf']


class DeepFool:
    def __init__(self, model, q=2, max_iter=50, args=None, **kwargs):
        """
        Args:
            q (int): p-1 / p, l_p norm to use in calculating distance
            max_iter (int): Number of iterations in perturbation
        """
        self.model = model
        self.max_iter = max_iter
        self.args = args
        self.num_classes = args.num_classes
        self.q = q
        self.eps = 10e-5

    def generate(self, images, labels):
        self.original_shape = images[0].shape
        adv_imgs = [self.generate_sample(image, label) for (image, label)
                    in zip(images, labels)]
        adv_imgs = torch.cat(adv_imgs, dim=0)

        return adv_imgs

    def generate_sample(self, image, label):
        """Fooling if label changes. Doesn't matter if source_label=true label or not
        """
        r_sum = 0
        niter = 0
        adv_img = image.clone().unsqueeze(0)
        adv_img.requires_grad_()

        output = self.model(adv_img)
        source_label = torch.max(output, 1)[1][0]

        adv_label = source_label
        while adv_label == source_label and niter < self.max_iter:
            source_grad = self.get_label_grad(adv_img, source_label)
            dist_min = np.inf 
            for k in range(self.num_classes):
                if k != source_label:
                    adv_output = self.model(adv_img)
                    target_grad = self.get_label_grad(adv_img, k)

                    w_k = target_grad - source_grad
                    f_k = adv_output[:, k] - adv_output[:, source_label]
                    dist = torch.abs(f_k) / w_k.norm(p=self.q)

                    if (dist < dist_min).all():  # torch bool value
                        dist_min = dist
                        w_min = w_k
                        f_min = f_k
                        l = k

            r_i = f_min.abs() / (w_min.norm(p=self.q) ** self.q) * w_min.abs().pow(self.q-1) * w_min.sign()
            adv_img = adv_img + r_i
            r_sum += r_i
            niter += 1
            adv_label = torch.max(self.model(adv_img), 1)[1][0]

        return image + r_sum

    def get_label_grad(self, image, label):
        image = image.clone()
        image.requires_grad_()
        image.retain_grad()

        output = self.model(image)
        grad_mask = torch.zeros_like(output)
        grad_mask[:, label] = 1

        output.backward(grad_mask, retain_graph=True)

        return image.grad


def deepfool_l2(model, args, **kwargs):
    return DeepFool(model, q=2, args=args, **kwargs)

def deepfool_l1(model, args, **kwargs):
    return DeepFool(model, q=np.inf, args=args, **kwargs)

def deepfool_linf(model, args, **kwargs):
    return DeepFool(model, q=1, args=args, **kwargs)
