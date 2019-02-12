import torch
import torch.nn.functional as F
import numpy as np

from common.summary import EvaluationMetrics
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
        if rcam:
            self.set_rcam()

    def generate(self, images):
        """
        Images (Tensor)
        """
        self.original_shape = images[0].shape

        def_imgs = [self.generate_sample(image) for image in images]
        def_imgs = torch.stack(def_imgs)

        return self.model(def_imgs)

    def generate_sample(self, image):
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

        args = self.args
        args.batch_size = 1
        train_loader, _ = get_loader(args)
        for i, (image, label) in enumerate(train_loader):
            if self.args.cuda:
                image = image.cuda()
                label = label.cuda()
            if self.args.half:
                image = image.half()

            weights = []
            with torch.no_grad():
                _ = self.model(image)
            weights = weights[0].squeeze()
            _, label = torch.max(label, dim=1)
            self.weights.update(label.item(), weights)
            if (i+1)%1000 == 0:
                print("{:5.1f}% ({}/{})".format((i+1)/len(train_loader)*100, i+1, len(train_loader)))
        print("Created CAM for {}".format(self.args.model))
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
        outputs = self.model(image.unsqueeze(0))
        outputs = outputs.detach().cpu().numpy().squeeze()
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
        rcam = rcam.squeeze()
        rcam = rcam.detach().cpu().numpy()

        return rcam


def pixeldeflection(model, args, **kwargs):
    return PixelDeflection(model, args=args)
