import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms

__all__ = ['randomization']


class Randomization:
    def __init__(self, model, scale=0.9, mode='nearest', pad_type='zero', args=None, **kwargs):
        self.model = model
        self.scale = scale
        self.mode = mode
        self.pad = self.get_padding(pad_type)
        self.args = args

    def generate(self, images):
        def_imgs = [self.generate_sample(image) for image in images]
        def_imgs = torch.stack(def_imgs)
        return self.model(def_imgs)

    def generate_sample(self, image):
        w = int(image.size(-2)*(1 - np.random.uniform()*(1 - self.scale)))
        h = int(image.size(-1)*(1 - np.random.uniform()*(1 - self.scale)))
        resized = self.resize(image, w, h)

        w_diff = image.size(-2) - resized.size(-2)
        h_diff = image.size(-1) - resized.size(-1)
        
        padded = self.pad(self.get_pad_shape(w_diff, h_diff))(resized)
        padded = padded.squeeze(0)
        
        return padded

    def get_padding(self, pad_type):
        if pad_type == "reflection":
            return nn.ReflectionPad2d
        elif pad_type == "replication":
            return nn.ReplicationPad2d
        elif pad_type == "zero":
            return nn.ZeroPad2d

    def get_pad_shape(self, w_diff, h_diff):
        w = np.random.choice(np.arange(-w_diff, w_diff+1))
        h = np.random.choice(np.arange(-h_diff, h_diff+1))
        left = int((w > 0)*w)
        up = int((h > 0)*h)
        return (left, w_diff - left, up, h_diff - up)

    def resize(self, image, w, h):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((h, w)),
            transforms.ToTensor()
        ])
        resized = transform(image.cpu())
        if self.args.cuda:
            resized = resized.cuda()
        return resized


def randomization(model, args, **kwargs):
    return Randomization(model, args=args, **kwargs)
