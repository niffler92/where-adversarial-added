import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from common.summary import EvaluationMetrics
from common.torch_utils import get_optimizer, to_var, to_np
from dataloader import normalize, denormalize, get_loader

__all__ = ['pixeldeflection']


class PixelDeflection:
    def __init__(self, model, ndeflection=100, window=10, sigma=0.04, denoiser='wavelet',
                 rcam=True, args=None, **kwargs):
        """
        Most of the code is from https://github.com/iamaaditya/pixel-deflection
        """
        self.model = model
        self.ndeflection = ndeflection
        self.window = window
        self.sigma = sigma
        self.denoiser = denoiser
        self.args = args
        if rcam: self.set_rcam()

    def generate(self, images, labels):
        """
        Images (Tensor)
        Labels (Tensor)
        """
        self.original_shape = images[0].shape

        def_imgs = [self.generate_sample(image, label) for (image, label)
                    in zip(images, labels)]
        def_imgs = torch.stack(def_imgs)
        def_outputs = self.model(to_var(def_imgs, volatile=True))
        def_probs, def_labels = torch.max(def_outputs, 1)

        return def_imgs, def_labels

    def generate_sample(self, image, label):
        rcam = self.get_rcam(image)
        def_image = self.pixel_deflection(image, rcam, self.ndeflection, self.window)
        def_image = denormalize(def_image.unsqueeze(0), self.args.dataset).squeeze(0)
        def_image = np.transpose(def_image.cpu().numpy(), [1, 2, 0])
        def_image = self.denoise(self.denoiser, def_image, self.sigma)
        def_image = np.transpose(def_image, [2, 0, 1])
        def_image = torch.FloatTensor(def_image).cuda()
        def_image = normalize(def_image.unsqueeze(0), self.args.dataset).squeeze(0)

        return def_image

    @staticmethod
    def pixel_deflection(img, rcam, ndeflection, window):
        C, H, W = img.shape
        while ndeflection > 0:
            for c in range(C):
                x,y = np.random.randint(0,H-1), np.random.randint(0,W-1)
                if np.random.uniform() < rcam[x,y]:
                    continue

                while True: #this is to ensure that PD pixel lies inside the image
                    a,b = np.random.randint(-1*window,window), np.random.randint(-1*window,window)
                    if x+a < H and x+a > 0 and y+b < W and y+b > 0: break
                img[c,x,y] = img[c,x+a,y+b]
                ndeflection -= 1
        return img

    @staticmethod
    def denoise(denoiser_name, img, sigma):
        from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, denoise_nl_means, wiener)
        if denoiser_name == 'wavelet':
            """Input scale - [0, 1]
            """
            return denoise_wavelet(img, sigma=sigma, mode='soft', multichannel=True, convert2ycbcr=True, method='BayesShrink')
        elif denoiser_name == 'TVM':
            return denoise_tv_chambolle(img, multichannel=True)
        elif denoiser_name == 'bilateral':
            return denoise_bilateral(img, bins=1000, multichannel=True)
        elif denoiser_name == 'deconv':
            return wiener(img)
        elif denoiser_name == 'NLM':
            return denoise_nl_means(img, multichannel=True)
        else:
            raise Exception('Incorrect denoiser mentioned. Options: wavelet, TVM, bilateral, deconv, NLM')

    def set_rcam(self):
        print("Creating CAM for {}".format(self.args.model))
        if 'resnet' in str.lower(type(self.model).__name__):
            last_conv = 'layer4'
        else:
            print("Model not implemented. Setting rcam=False by default.")
            return

        self.weights = EvaluationMetrics(list(range(self.args.num_classes)))
        def hook_weights(module, input, output):
            weights.append(F.adaptive_max_pool2d(output, (1,1)))
        handle = self.model._modules.get(last_conv).register_forward_hook(hook_weights)

        train_loader, _ = get_loader(self.args.dataset,
            batch_size=1,
            num_workers=self.args.workers
        )
        for i, (image, label) in enumerate(train_loader):
            weights = []
            _ = self.model(to_var(image, volatile=True))
            weights = weights[0].squeeze()
            label = label.squeeze()[0]
            self.weights.update(label, weights)
            if (i+1)%1000 == 0:
                print("{:5.1f}% ({}/{})".format((i+1)/len(train_loader)*100, i+1, len(train_loader)))
        handle.remove()

    def get_rcam(self, image, k=1):
        size = image.shape[-2:]
        if not hasattr(self, 'weights'):
            return torch.zeros(size)
        if 'resnet' in str.lower(type(self.model).__name__):
            last_conv = 'layer4'
        else:
            return torch.zeros(size)

        features = []
        def hook_feature(module, input, output):
            features.append(output)
        handle = self.model._modules.get(last_conv).register_forward_hook(hook_feature)
        outputs = self.model(to_var(image.unsqueeze(0), volatile=True))
        outputs = to_np(outputs).squeeze()
        handle.remove()

        features = features[0]
        weights = self.weights.avg

        _, nc, h, w = features.shape
        cams = []
        for label in range(self.args.num_classes):
            cam = weights[label]@features.view(nc, h*w)
            cam = cam.view(h, w)
            cam = (cam - torch.min(cam))/(torch.max(cam) - torch.min(cam))
            cam = cam.view(1,1,*cam.shape)
            cams.append(F.upsample(cam, size, mode='bilinear'))
        rcam = 0
        for idx, label in enumerate(np.argsort(outputs)):
            if idx >= k:
                break
            else:
                rcam += cams[label]/float(2**(idx+1))
        rcam = (rcam - torch.min(rcam))/(torch.max(rcam) - torch.min(rcam))
        rcam = to_np(rcam).squeeze()

        return rcam


def pixeldeflection(model, args, **kwargs):
    return PixelDeflection(model, ndeflection=args.ndeflection, window=args.window, sigma=args.sigma, args=args)
