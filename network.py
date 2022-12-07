import torch
import torch.nn as nn
from modules import *


class DiffusionNetwork(nn.Module):
    def __init__(self, in_size, t_range, img_depth):
        super().__init__()
        self.t_range = t_range
        self.in_size = in_size

        bilinear = True
        self.inc = DoubleConv(img_depth, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down3 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, img_depth)
        self.sa1 = SAWrapper(128, 16)
        self.sa3 = SAWrapper(64, 16)

    def pos_encoding(self, t, channels, embed_size):
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2).to(t).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc.view(-1, channels, 1, 1).repeat(1, 1, embed_size, embed_size)

    def forward(self, x, t):
        """
        Model is U-Net with added positional encodings and self-attention layers.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1) + self.pos_encoding(t, 128, 16)
        x2 = self.sa1(x2)
        x3 = self.down2(x2) + self.pos_encoding(t, 256, 8)
        x4 = self.down3(x3) + self.pos_encoding(t, 256, 4)
        x = self.up1(x4, x3) + self.pos_encoding(t, 128, 8)
        x = self.up2(x, x2) + self.pos_encoding(t, 64, 16)
        x = self.sa3(x)
        x = self.up3(x, x1) + self.pos_encoding(t, 64, 32)
        output = self.outc(x)
        return output
