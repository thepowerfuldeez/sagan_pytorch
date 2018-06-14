import torch
import numpy as np

from torch import nn
from spectral_normalization import SpectralNorm
from self_attention import SelfAttention


class Generator(nn.Module):
    """Generator."""

    def __init__(self, batch_size, image_size=64, z_dim=100, conv_dim=64):
        super(Generator, self).__init__()
        self.imsize = image_size

        repeat_num = int(np.log2(self.imsize)) - 3
        mult = 2 ** repeat_num  # 8
        self.l1 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(z_dim, conv_dim * mult, 4)),
            nn.BatchNorm2d(conv_dim * mult),
            nn.ReLU()
        )

        curr_dim = conv_dim * mult
        self.l2 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, 2, 1)),
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.ReLU()
        )

        curr_dim //= 2
        self.l3 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, 2, 1)),
            nn.BatchNorm2d(curr_dim // 2),
            nn.ReLU()
        )

        if self.imsize == 64:
            curr_dim //= 2
            self.l4 = nn.Sequential(
                SpectralNorm(nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, 2, 1)),
                nn.BatchNorm2d(curr_dim // 2),
                nn.ReLU()
            )
            curr_dim //= 2

        self.last = nn.Sequential(
            nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1),
            nn.Tanh()
        )

        self.attn1 = SelfAttention(128, 'relu')
        self.attn2 = SelfAttention(64, 'relu')

    def forward(self, z):
        # from 1d conv
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.l1(z)
        out = self.l2(out)
        out = self.l3(out)
        out, p1 = self.attn1(out)
        out = self.l4(out)
        out, p2 = self.attn2(out)
        out = self.last(out)

        return out, p1, p2


class Discriminator(nn.Module):
    def __init__(self, batch_size=64, image_size=64, conv_dim=64):
        super(Discriminator, self).__init__()
        self.imsize = image_size

        self.l1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(3, conv_dim, 4, 2, 1)),
            nn.LeakyReLU(0.1)
        )
        curr_dim = conv_dim

        self.l2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)),
            nn.LeakyReLU(0.1)
        )
        curr_dim *= 2

        self.l3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)),
            nn.LeakyReLU(0.1)
        )
        curr_dim *= 2

        if self.imsize == 64:
            self.l4 = nn.Sequential(
                SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)),
                nn.LeakyReLU(0.1)
            )
            curr_dim *= 2
        self.last = nn.Sequential(nn.Conv2d(curr_dim, 1, 4))

        self.attn1 = SelfAttention(256, 'relu')
        self.attn2 = SelfAttention(512, 'relu')

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out, p1 = self.attn1(out)
        out = self.l4(out)
        out, p2 = self.attn2(out)
        out = self.last(out)

        return out.squeeze(), p1, p2
