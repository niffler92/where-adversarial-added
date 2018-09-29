import torch
import torch.nn as nn
import numpy as np

from common.torch_utils import to_var


class SplitConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(SplitConv2d, self).__init__()
        self.stride = stride
        if kernel_size % stride == 0:
            self.overlap = 1
        else:
            self.overlap = int(kernel_size/stride + 1)
        for i in range(self.overlap**2):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                        stride=stride*self.overlap,
                                        padding=padding,
                                        bias=False)
            if torch.cuda.is_available():
                conv = conv.cuda()
            self.add_module('split'+str(i+1), conv)

    def forward(self, input):
        shape = self.split1(input).shape
        output = to_var(torch.zeros(shape).repeat(1,1,self.overlap,self.overlap))

        for i, conv in enumerate(self._modules.values()):
            out = input
            col = int(i % self.overlap)
            if col != 0:
                col_padding = to_var(torch.zeros(out.size(2), 1))
                out = torch.cat((col_padding.repeat(out.size(0), out.size(1), 1, col*self.stride), out), dim=3)
            row = int(i / self.overlap)
            if row != 0:
                row_padding = to_var(torch.zeros(1, out.size(3)))
                out = torch.cat((row_padding.repeat(out.size(0), out.size(1), row*self.stride, 1), out), dim=2)
            out = conv(out)
            if out.size(2) != shape[2]:
                out = out[:,:,1:,:].contiguous()
            if out.size(3) != shape[3]:
                out = out[:,:,:,1:].contiguous()

            out = out.view(shape[0], shape[1], 1, -1)
            out = out.expand(-1,-1,self.overlap,-1).transpose(-2,-1)
            out = out.contiguous().view(shape[0], shape[1], shape[2]*self.overlap, -1)
            out = out.view(shape[0], shape[1], 1, -1)
            out = out.expand(-1,-1,self.overlap,-1)
            out = out.split(shape[2]*self.overlap, dim=3)
            out = torch.cat(out, dim=2)

            mask = np.zeros((self.overlap, self.overlap))
            mask[row, col] = 1
            mask = np.vstack([np.hstack([mask]*shape[3])]*shape[2])
            mask = to_var(torch.FloatTensor(mask))
            mask = mask.view(1, 1, mask.size(0), -1)

            out = out*mask
            output += out

        return output
