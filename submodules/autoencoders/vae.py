import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['vae_v0', 'vae_v1', 'vae_v2', 'vae_v3', 'vae_v4', 'vae_v5']


class ResBlock(nn.Module):
    def __init__(self, in_channels, d):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, kernel_size=1, stride=1, padding=0)
        )
        
    def forward(self, x):
        return x + self.convs(x)


class VAE(nn.Module):
    def __init__(self, in_channels=3, d=128, config=0, args=None, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, d//2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d//2, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            
            ResBlock(d, d),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d),
        )
        self.decoder = nn.Sequential(
            ResBlock(d, d),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d),
            nn.BatchNorm2d(d),

            nn.ConvTranspose2d(d, d//2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d//2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d//2, d//2, kernel_size=4, stride=2, padding=1, bias=False)
        )
        
        if config == 0:
            self.first_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=2)
            self.deconv = nn.ConvTranspose2d(d//2, d//2, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.f1 = 4
            self.f2 = 4
        elif config == 1:
            self.first_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=(2,1))
            self.deconv = nn.ConvTranspose2d(d//2, d//2, kernel_size=3, stride=(2,1), padding=1, output_padding=(1,0))
            self.f1 = 4
            self.f2 = 8
        elif config == 2:
            self.first_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=(1,2))
            self.deconv = nn.ConvTranspose2d(d//2, d//2, kernel_size=3, stride=(1,2), padding=1, output_padding=(0,1))
            self.f1 = 8
            self.f2 = 4
        elif config == 3:
            self.first_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1,2), stride=(2,4))
            self.deconv = nn.ConvTranspose2d(d//2, d//2, kernel_size=3, stride=(2,4), padding=1, output_padding=(1,3))
            self.f1 = 4
            self.f2 = 2
        elif config == 4:
            self.first_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(2,1), stride=(4,2))
            self.deconv = nn.ConvTranspose2d(d//2, d//2, kernel_size=3, stride=(4,2), padding=1, output_padding=(3,1))
            self.f1 = 2
            self.f2 = 4
        elif config == 5:
            self.first_conv = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=4)
            self.deconv = nn.ConvTranspose2d(d//2, d//2, kernel_size=3, stride=4, padding=1, output_padding=3)
            self.f1 = 2
            self.f2 = 2

        self.d = d
        if args.dataset == "ImageNet":
            self.f1 *= 7
            self.f2 *= 7
        self.fc1 = nn.Linear(d*self.f1*self.f2, d*self.f1*self.f2)
        self.fc2 = nn.Linear(d*self.f1*self.f2, d*self.f1*self.f2)

        self.last_conv = nn.Conv2d(d//2, in_channels, kernel_size=3, padding=1)

    def encode(self, x):
        h1 = self.encoder(x)
        h1 = h1.view(h1.size(0), -1)
        return self.fc1(h1), self.fc2(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = z.view(-1, self.d, self.f1, self.f2)
        h3 = self.decoder(z)
        return F.tanh(h3)

    def forward(self, x):
        x = self.first_conv(x)
        
        self.mu, self.logvar = self.encode(x)
        z = self.reparameterize(self.mu, self.logvar)
        out = self.decode(z)

        out = self.deconv(out)
        out = self.last_conv(out)

        return out


def vae_v0(args, **kwargs):
    return VAE(args=args, **kwargs)

def vae_v1(args, **kwargs):
    return VAE(config=1, args=args, **kwargs)

def vae_v2(args, **kwargs):
    return VAE(config=2, args=args, **kwargs)

def vae_v3(args, **kwargs):
    return VAE(config=3, args=args, **kwargs)

def vae_v4(args, **kwargs):
    return VAE(config=4, args=args, **kwargs)

def vae_v5(args, **kwargs):
    return VAE(config=5, args=args, **kwargs)
