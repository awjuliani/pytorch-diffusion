import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
from modules import *
from network import DiffusionNetwork


class DiffusionProcess(pl.LightningModule):
    def __init__(self, in_size, t_range, img_depth, num_classes, guidance_scale):
        super().__init__()
        self.t_range = t_range
        self.in_size = in_size
        self.img_depth = img_depth
        self.num_classes = num_classes
        self.guidance_scale = guidance_scale
        self.network = DiffusionNetwork(in_size, t_range, img_depth, num_classes)

    def forward(self, x, t, y):
        xt = self.network(x, t, y)
        return xt

    def func(self, t):
        s = 0.008
        num = t / self.t_range + s
        den = 1 + s
        mul = math.pi / 2
        return torch.pow(torch.cos(num / den * mul), 2)

    def beta(self, t):
        base = 1 - self.alpha_bar(t) / self.alpha_bar(t - 1)
        return torch.clip(base, 0.0, 0.999)

    def alpha(self, t):
        return 1 - self.beta(t)

    def alpha_bar(self, t):
        return self.func(t) / self.func(torch.zeros_like(t))

    def get_loss(self, batch, batch_idx):
        """
        Corresponds to Algorithm 1 from (Ho et al., 2020).
        """
        x, y = batch
        y = torch.nn.functional.one_hot(y, self.num_classes).type(torch.float)
        y[y.shape[0] // 2 + y.shape[0] // 4 :] = 0
        ts = torch.randint(1, self.t_range, [x.shape[0]], device=self.device)
        epsilons = torch.randn(x.shape, device=self.device)
        a_hats = self.alpha_bar(ts).view(-1, 1, 1, 1).to(self.device)
        noise_imgs = torch.sqrt(a_hats) * x + torch.sqrt(1 - a_hats) * epsilons
        ts = ts.unsqueeze(-1).type(torch.float)
        e_hat = self.forward(noise_imgs, ts, y)
        loss = nn.functional.mse_loss(
            e_hat.reshape(-1, self.in_size), epsilons.reshape(-1, self.in_size)
        )
        return loss

    def diffusion_step(self, x, t, y):
        """
        Corresponds to the inner loop of Algorithm 2 from (Ho et al., 2020).
        """
        with torch.no_grad():
            if t > 1:
                z = torch.randn(x.shape).to(x)
            else:
                z = 0
            e_hat_a = self.forward(x, t.view(1, 1).repeat(x.shape[0], 1), y)
            e_hat_b = self.forward(x, t.view(1, 1).repeat(x.shape[0], 1), y * 0)
            e_hat = (1 + self.guidance_scale) * e_hat_a - self.guidance_scale * e_hat_b
            pre_scale = 1 / math.sqrt(self.alpha(t))
            e_scale = (1 - self.alpha(t)) / math.sqrt(1 - self.alpha_bar(t))
            post_sigma = math.sqrt(self.beta(t)) * z
            x = pre_scale * (x - e_scale * e_hat) + post_sigma
            return x

    def diffuse_sample(self, batch_size, n_hold_final=0):
        """
        Corresponds to Algorithm 2 from (Ho et al., 2020).
        """
        with torch.no_grad():
            gen_samples = []
            x = torch.randn(
                (batch_size, self.img_depth, self.in_size, self.in_size)
            ).to(self.device)
            y = torch.arange(0, 9).to(self.device)
            y = torch.nn.functional.one_hot(y, self.num_classes).type(torch.float)
            ts = torch.arange(self.t_range - 1, 0, -1).to(self.device)
            for t in ts:
                x = self.diffusion_step(x, t, y)
                if t % 50 == 0:
                    gen_samples.append(x)
            for _ in range(n_hold_final):
                gen_samples.append(x)
            gen_samples = torch.stack(gen_samples, dim=0).moveaxis(2, 4).squeeze(-1)
            gen_samples = (gen_samples.clamp(-1, 1) + 1) / 2
            return gen_samples

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)
        self.log("val/loss", loss)
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return optimizer
