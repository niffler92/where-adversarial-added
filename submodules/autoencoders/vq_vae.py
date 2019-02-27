import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

__all__ = ['vq_vae_v0', 'vq_vae_v1', 'vq_vae_v2', 'vq_vae_v3', 'vq_vae_v4', 'vq_vae_v5']


class NearestEmbedFunc(Function):
    @staticmethod
    def forward(ctx, input, emb):
        # save sizes for backward
        ctx.batch_size = input.size(0)
        ctx.num_latents = int(np.prod(np.array(input.size()[2:])))
        ctx.emb_dim = emb.size(0)
        ctx.num_emb = emb.size(1)
        ctx.input_type = type(input)
        ctx.dims = list(range(len(input.size())))

        # expand so it is broadcastable
        x_expanded = input.unsqueeze(-1)
        num_arbitrary_dims = len(ctx.dims) - 2
        if num_arbitrary_dims:
            emb_expanded = emb.view(emb.shape[0], *([1]*num_arbitrary_dims), emb.shape[1])
        else:
            emb_expanded = emb

        # find nearest neighbors
        dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        _, argmin = dist.min(-1)
        shifted_shape = [input.shape[0], *list(input.shape[2:]), input.shape[1]]
        result = emb.t().index_select(0, argmin.view(-1)).view(shifted_shape)
        result = result.permute(0, ctx.dims[-1], *ctx.dims[1:-1]).contiguous()

        ctx.argmin = argmin
        return result, argmin

    @staticmethod
    def backward(ctx, grad_output, argmin=None):
        grad_input = grad_emb = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output
        if ctx.needs_input_grad[1]:
            latent_indices = torch.arange(ctx.num_emb).type_as(ctx.argmin)
            idx_choices = (ctx.argmin.view(-1,1) == latent_indices.view(1,-1)).float()
            n_idx_choice = idx_choices.type_as(grad_output).sum(0)
            n_idx_choice[n_idx_choice == 0] = 1
            idx_avg_choices = idx_choices / n_idx_choice
            grad_output = grad_output.permute(0, *ctx.dims[2:], 1).contiguous()
            grad_output = grad_output.view(ctx.batch_size*ctx.num_latents, ctx.emb_dim)
            grad_emb = torch.sum(
                grad_output.view(-1, ctx.emb_dim, 1)*idx_avg_choices.view(-1, 1, ctx.num_emb), 0)

        return grad_input, grad_emb, None, None


class NearestEmbed(nn.Module):
    def __init__(self, num_embeddings, embeddings_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(embeddings_dim, num_embeddings))

    def forward(self, x):
        return NearestEmbedFunc().apply(x, self.weight)


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


class VQ_VAE(nn.Module):
    def __init__(self, in_channels=3, d=128, k=10, config=0, args=None, **kwargs):
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
        self.emb = NearestEmbed(k, d)

        if config == 0:
            self.first_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=2)
            self.deconv = nn.ConvTranspose2d(d//2, d//2, kernel_size=3, stride=2, padding=1, output_padding=1)
        elif config == 1:
            self.first_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=(2,1))
            self.deconv = nn.ConvTranspose2d(d//2, d//2, kernel_size=3, stride=(2,1), padding=1, output_padding=(1,0))
        elif config == 2:
            self.first_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=(1,2))
            self.deconv = nn.ConvTranspose2d(d//2, d//2, kernel_size=3, stride=(1,2), padding=1, output_padding=(0,1))
        elif config == 3:
            self.first_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1,2), stride=(2,4))
            self.deconv = nn.ConvTranspose2d(d//2, d//2, kernel_size=3, stride=(2,4), padding=1, output_padding=(1,3))
        elif config == 4:
            self.first_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(2,1), stride=(4,2))
            self.deconv = nn.ConvTranspose2d(d//2, d//2, kernel_size=3, stride=(4,2), padding=1, output_padding=(3,1))
        elif config == 5:
            self.first_conv = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=4)
            self.deconv = nn.ConvTranspose2d(d//2, d//2, kernel_size=3, stride=4, padding=1, output_padding=3)

        self.last_conv = nn.Conv2d(d//2, in_channels, kernel_size=3, padding=1)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return F.tanh(self.decoder(x))

    def forward(self, x):
        x = self.first_conv(x)
        
        self.z_e = self.encode(x)
        z_q, _ = self.emb(self.z_e)
        self.z_q, _ = self.emb(self.z_e.detach())
        out = self.decode(z_q)

        out = self.deconv(out)
        out = self.last_conv(out)

        return out


def vq_vae_v0(args, **kwargs):
    return VQ_VAE(args=args, **kwargs)

def vq_vae_v1(args, **kwargs):
    return VQ_VAE(config=1, args=args, **kwargs)

def vq_vae_v2(args, **kwargs):
    return VQ_VAE(config=2, args=args, **kwargs)

def vq_vae_v3(args, **kwargs):
    return VQ_VAE(config=3, args=args, **kwargs)

def vq_vae_v4(args, **kwargs):
    return VQ_VAE(config=4, args=args, **kwargs)

def vq_vae_v5(args, **kwargs):
    return VQ_VAE(config=5, args=args, **kwargs)
