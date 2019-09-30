import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from common.torch_utils import to_var

__all__ = ['randomization']


class Randomization:
    def __init__(self, model, scale=0.9, mode='nearest', pad_type='zero', args=None, **kwargs):
        self.model = model
        self.scale = scale
        self.mode = mode
        self.pad = self.get_padding(pad_type)
        self.args = args

    def generate(self, images, labels):
        def_imgs = [self.generate_sample(image, label) for (image, label)
                    in zip(images, labels)]
        def_imgs = torch.stack(def_imgs)
        def_outputs = self.model(to_var(def_imgs))
        def_probs, def_labels = torch.max(def_outputs, 1)

        return def_imgs, def_labels

    def generate_sample(self, image, label):
        scale = 1 - np.random.uniform()*(1 - self.scale)
        resized = self.resize(image.unsqueeze(0), scale_factor=scale, mode=self.mode)
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

    @staticmethod
    def resize(x, size=None, scale_factor=None, mode='nearest'):
    	# define size if user has specified scale_factor
    	if size is None: size = (int(scale_factor*x.size(2)), int(scale_factor*x.size(3)))
    	# create coordinates
    	h = torch.arange(0,size[0]) / (size[0]-1) * 2 - 1
    	w = torch.arange(0,size[1]) / (size[1]-1) * 2 - 1
    	# create grid
    	grid = torch.zeros(size[0],size[1],2)
    	grid[:,:,0] = w.unsqueeze(0).repeat(size[0],1)
    	grid[:,:,1] = h.unsqueeze(0).repeat(size[1],1).transpose(0,1)
    	# expand to match batch size
    	grid = grid.unsqueeze(0).repeat(x.size(0),1,1,1)
    	if x.is_cuda: grid = grid.cuda()
    	# do sampling
    	return F.grid_sample(x, grid, mode=mode)


def randomization(model, args, **kwargs):
    return Randomization(model, args=args, **kwargs)
