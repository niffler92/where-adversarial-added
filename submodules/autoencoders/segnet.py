import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['segnet']


class SegNet(nn.Module):
    def __init__(self, in_channels=3, args=None, **kwargs):
        super(SegNet, self).__init__()

        self.layer1p = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
            )
        self.layer2p = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
            )
        self.layer3p = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
            )
        self.layer4p = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
            )
        self.layer5p = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
            )

        self.layer5d = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
            )
        self.layer4d = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
            )
        self.layer3d = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
            )
        self.layer2d = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
            )
        self.layer1d = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
            )

    def forward(self, x):
        out = self.layer1p(x)
        x1p, id1 = F.max_pool2d(out, kernel_size=2, stride=2, return_indices=True)
        out = self.layer2p(x1p)
        x2p, id2 = F.max_pool2d(out, kernel_size=2, stride=2, return_indices=True)
        out = self.layer3p(x2p)
        x3p, id3 = F.max_pool2d(out, kernel_size=2, stride=2, return_indices=True)
        out = self.layer4p(x3p)
        x4p, id4 = F.max_pool2d(out, kernel_size=2, stride=2, padding=1, return_indices=True)
        out = self.layer5p(x4p)
        x5p, id5 = F.max_pool2d(out, kernel_size=2, stride=2, return_indices=True)

        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2)
        out = self.layer5d(x5d)
        x4d = F.max_unpool2d(out, id4, kernel_size=2, stride=2, padding=1)
        out = self.layer4d(x4d)
        x3d = F.max_unpool2d(out, id3, kernel_size=2, stride=2)
        out = self.layer3d(x3d)
        x2d = F.max_unpool2d(out, id2, kernel_size=2, stride=2)
        out = self.layer2d(x2d)
        x1d = F.max_unpool2d(out, id1, kernel_size=2, stride=2)
        out = self.layer1d(x1d)

        return out


def segnet(args, **kwargs):
    return SegNet(args=args, **kwargs)
