import torch
import torch.nn as nn
from modules import *


class DiffusionNetwork(nn.Module):
    def __init__(self, in_size, t_range, img_depth, num_classes):
        super().__init__()
        self.t_range = t_range
        self.in_size = in_size
        self.num_classes = num_classes

        bilinear = True
        self.proj_y = nn.Sequential(nn.Linear(num_classes, 32), nn.GELU())
        self.inc = DoubleConv(img_depth + 32, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down3 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, img_depth)
        self.sa_down = SAWrapper(128, 16)
        self.sa_up = SAWrapper(64, 16)

    def positional_encoding(self, t, embed_dim, embed_height_width):
        inv_freq_a = 1.0 / (
            10000 ** (torch.arange(0, embed_dim, 2).to(t).float() / embed_dim)
        )
        inv_freq_b = 1.0 / (
            10000 ** (torch.arange(1, embed_dim, 2).to(t).float() / embed_dim)
        )
        pos_enc_a = torch.sin(t.repeat(1, embed_dim // 2) * inv_freq_a)
        pos_enc_b = torch.cos(t.repeat(1, embed_dim // 2) * inv_freq_b)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc.view(-1, embed_dim, 1, 1).repeat(
            1, 1, embed_height_width, embed_height_width
        )

    def forward(self, x, t, y):
        """
        Model is U-Net with added positional encodings and self-attention layers.
        """
        y = self.proj_y(y)
        y = y.view(-1, 32, 1, 1).repeat(1, 1, self.in_size, self.in_size)
        x = torch.cat([x, y], dim=1)

        x1 = self.inc(x) + self.positional_encoding(t, 64, 32)
        x2 = self.down1(x1) + self.positional_encoding(t, 128, 16)
        x2 = self.sa_down(x2)
        x3 = self.down2(x2) + self.positional_encoding(t, 256, 8)
        x4 = self.down3(x3) + self.positional_encoding(t, 256, 4)
        x = self.up1(x4, x3) + self.positional_encoding(t, 128, 8)
        x = self.up2(x, x2) + self.positional_encoding(t, 64, 16)
        x = self.sa_up(x)
        x = self.up3(x, x1) + self.positional_encoding(t, 64, 32)
        output = self.outc(x)
        return output
