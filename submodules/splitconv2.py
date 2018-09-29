import torch
import torch.nn as nn


class SplitConv2d(nn.Module):
    """2x2, split and concat and 1x1 conv
    in_channel = out_chanenl
    """
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=1, padding=0, bias=False):
        super().__init__()
        self.stride = stride

        if stride == 1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False)
            self.bn = nn.BatchNorm2d(out_channels, affine=True)
        elif stride == 2:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, bias=bias)
            #self.bn1 = nn.BatchNorm2d(out_channels, affine=True)
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=(1,0), bias=bias)
            #self.bn2 = nn.BatchNorm2d(out_channels, affine=True)
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=(0,1), bias=bias)
            #self.bn3 = nn.BatchNorm2d(out_channels, affine=True)
            self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=(1,1), bias=bias)
            #self.bn4 = nn.BatchNorm2d(out_channels, affine=True)

            self.conv = nn.Conv2d(4*out_channels, out_channels, 1, 1, bias=bias)  # 1x1 conv
            self.bn = nn.BatchNorm2d(out_channels, affine=True)

            self.activation = nn.ReLU()


    def forward(self, x):
        if self.stride == 1:
            out = self.bn(self.conv(x))
        elif self.stride == 2:
            #out1 = self.activation(self.bn1(self.conv1(x)))
            #out2 = self.activation(self.bn2(self.conv2(x)[:, :, 1:, :].contiguous()))
            #out3 = self.activation(self.bn3(self.conv3(x)[:, :, :, 1:].contiguous()))
            #out4 = self.activation(self.bn4(self.conv4(x)[:, :, 1:, 1:].contiguous()))
            out1 = self.conv1(x)
            out2 = self.conv2(x)[:, :, 1:, :].contiguous()
            out3 = self.conv3(x)[:, :, :, 1:].contiguous()
            out4 = self.conv4(x)[:, :, 1:, 1:].contiguous()
            out = torch.cat([out1, out2, out3, out4], dim=1)
            out = self.bn(self.conv(out))
        return out


class SplitConv2d_2(nn.Module):
    """2x2, split and concat and 1x1 conv
    in_channel = out_chanenl
    """
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=1, padding=0, bias=False):
        super().__init__()
        self.stride = stride

        if stride == 1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False)
            self.bn = nn.BatchNorm2d(out_channels, affine=True)
        elif stride == 2:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, bias=bias)
            self.bn1 = nn.BatchNorm2d(out_channels, affine=True)
            self.conv1_2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, bias=False, padding=1)
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=(1,0), bias=bias)
            self.bn2 = nn.BatchNorm2d(out_channels, affine=True)
            self.conv2_2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, bias=False, padding=1)
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=(0,1), bias=bias)
            self.bn3 = nn.BatchNorm2d(out_channels, affine=True)
            self.conv3_2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, bias=False, padding=1)
            self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=(1,1), bias=bias)
            self.bn4 = nn.BatchNorm2d(out_channels, affine=True)
            self.conv4_2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, bias=False, padding=1)

            self.conv = nn.Conv2d(4*out_channels, out_channels, 1, 1, bias=bias)  # 1x1 conv
            self.bn = nn.BatchNorm2d(out_channels, affine=True)

            self.activation = nn.ReLU()


    def forward(self, x):
        if self.stride == 1:
            out = self.bn(self.conv(x))
        elif self.stride == 2:
            out1 = self.conv1_2(self.activation(self.bn1(self.conv1(x))))
            out2 = self.conv2_2(self.activation(self.bn2(self.conv2(x)[:, :, 1:, :].contiguous())))
            out3 = self.conv3_2(self.activation(self.bn3(self.conv3(x)[:, :, :, 1:].contiguous())))
            out4 = self.conv4_2(self.activation(self.bn4(self.conv4(x)[:, :, 1:, 1:].contiguous())))
            #print(out1.shape)
            #print(out2.shape)
            #print(out3.shape)
            #print(out4.shape)
            out = torch.cat([out1, out2, out3, out4], dim=1)
            out = self.bn(self.conv(out))
        return out


class SplitConv2d_3(nn.Module):
    """2x2, split and concat and 1x1 conv
    """
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=1, padding=0, bias=False):
        super().__init__()
        self.stride = stride

        if stride == 1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False)
            self.bn = nn.BatchNorm2d(out_channels, affine=True)
        elif stride == 2:
            self.conv1 = nn.Conv2d(in_channels, int(out_channels/4), kernel_size, stride=2, bias=bias)
            self.bn1 = nn.BatchNorm2d(int(out_channels/4), affine=True)
            self.conv1_2 = nn.Conv2d(int(out_channels/4), int(out_channels/4), 3, stride=1, bias=False, padding=1)
            self.conv2 = nn.Conv2d(in_channels, int(out_channels/4), kernel_size, stride=2, padding=(1,0), bias=bias)
            self.bn2 = nn.BatchNorm2d(int(out_channels/4), affine=True)
            self.conv2_2 = nn.Conv2d(int(out_channels/4), int(out_channels/4), 3, stride=1, bias=False, padding=1)
            self.conv3 = nn.Conv2d(in_channels, int(out_channels/4), kernel_size, stride=2, padding=(0,1), bias=bias)
            self.bn3 = nn.BatchNorm2d(int(out_channels/4), affine=True)
            self.conv3_2 = nn.Conv2d(int(out_channels/4), int(out_channels/4), 3, stride=1, bias=False, padding=1)
            self.conv4 = nn.Conv2d(in_channels, int(out_channels/4), kernel_size, stride=2, padding=(1,1), bias=bias)
            self.bn4 = nn.BatchNorm2d(int(out_channels/4), affine=True)
            self.conv4_2 = nn.Conv2d(int(out_channels/4), int(out_channels/4), 3, stride=1, bias=False, padding=1)


        self.bn = nn.BatchNorm2d(out_channels, affine=True)
        self.activation = nn.ReLU()

    def forward(self, x):
        if self.stride == 1:
            out = self.activation(self.bn(self.conv(x)))
        elif self.stride == 2:
            out1 = self.conv1_2(self.activation(self.bn1(self.conv1(x))))
            out2 = self.conv2_2(self.activation(self.bn2(self.conv2(x)[:, :, 1:, :].contiguous())))
            out3 = self.conv3_2(self.activation(self.bn3(self.conv3(x)[:, :, :, 1:].contiguous())))
            out4 = self.conv4_2(self.activation(self.bn4(self.conv4(x)[:, :, 1:, 1:].contiguous())))
            out = torch.cat([out1, out2, out3, out4], dim=1)
            out = self.activation(self.bn(out))
        return out
