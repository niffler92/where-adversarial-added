import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from common.torch_utils import get_optimizer, to_var

__all__ = ['pixelshift']


class PixelShift:
    def __init__(self, model, x=1, y=0, is_random=False, pad_type="zero", args=None, **kwargs):
        """
        Shift an image after padding
        Args:
            x (int): shifts image to amount given at x coordinate (+ is shift right)
            y (int): shifts image to amount given at y coordinate (+ is shift up)
            is_random (bool): if true, random padding else pad with x, y coordinate
            pad_type (str): type of padding to do for shifting
        """
        assert pad_type in ["reflection", "replication", "zero"]
        self.model = model
        self.x = x
        self.y = y
        self.is_random = is_random
        self.pad_type = pad_type
        self.args = args

    def generate(self, images, labels):
        # XXX: input should be type Variable in order to calculate gradient for EOT... (graph)
        left, right, up, down = self.get_pad_shape(self.is_random)

        pad = self.get_padding(self.pad_type)(padding=(left, right, up, down))
        img_padded = pad(images)
        width, height = img_padded.shape[-1], img_padded.shape[-2]
        img_shifted = img_padded[:, :, down:height-up, right:width-left].contiguous()

        def_outputs = self.model(img_shifted)
        def_probs, def_labels = torch.max(def_outputs, 1)

        def_images = img_shifted if isinstance(images, Variable) else img_shifted.data

        return def_images, def_labels

    def get_pad_shape(self, is_random):
        if is_random:
            choices = [(1, 0), (0, 1), (1, 1), (-1, 0), (0, -1), (1, -1), (-1, 1), (-1, -1)]  # (0, 0)
            if self.args.attack == 'eot':
                choices += [(0, 0)]
            self.x, self.y = choices[np.random.choice(range(len(choices)), size=1)[0]]

        if self.x > 0:
            left = self.x
            right = 0
        else:
            left = 0
            right = -self.x

        if self.y > 0:
            up = 0
            down = self.y
        else:
            up = -self.y
            down = 0

        return (left, right, up, down)

    def get_padding(self, pad_type):
        if pad_type == "reflection":
            return nn.ReflectionPad2d
        elif pad_type == "replication":
            return nn.ReplicationPad2d
        elif pad_type == "zero":
            return nn.ZeroPad2d


def pixelshift(model, args, **kwargs):
    return PixelShift(model, x=args.x_coord, y=args.y_coord, is_random=args.random, pad_type=args.pad_type, args=args)
