import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
from modules import *
from network import DiffusionNetwork


class DiffusionProcess(pl.LightningModule):
    def __init__(self, in_size, t_range, img_depth):
        super().__init__()
        self.t_range = t_range
        self.in_size = in_size
        self.network = DiffusionNetwork(in_size, t_range, img_depth)

    def forward(self, x, t):
        xt = self.network(x, t)
        return xt

    def func(self, t):
        s = 0.008
        num = t / self.t_range + s
        den = 1 + s
        mul = math.pi / 2
        return math.pow(math.cos(num / den * mul), 2)

    def beta(self, t):
        base = torch.tensor(1 - self.alpha_bar(t) / self.alpha_bar(t - 1))
        return torch.clip(base, 0.0, 0.999)

    def alpha(self, t):
        return 1 - self.beta(t)

    def alpha_bar(self, t):
        return self.func(t) / self.func(0)

    def get_loss(self, batch, batch_idx):
        """
        Corresponds to Algorithm 1 from (Ho et al., 2020).
        """
        ts = torch.randint(0, self.t_range, [batch.shape[0]], device=self.device)
        noise_imgs = []
        epsilons = torch.randn(batch.shape, device=self.device)
        for i in range(len(ts)):
            a_hat = self.alpha_bar(ts[i])
            noise_imgs.append(
                (math.sqrt(a_hat) * batch[i]) + (math.sqrt(1 - a_hat) * epsilons[i])
            )
        noise_imgs = torch.stack(noise_imgs, dim=0)
        e_hat = self.forward(noise_imgs, ts.unsqueeze(-1).type(torch.float))
        loss = nn.functional.mse_loss(
            e_hat.reshape(-1, self.in_size), epsilons.reshape(-1, self.in_size)
        )
        return loss

    def denoise_sample(self, x, t):
        """
        Corresponds to the inner loop of Algorithm 2 from (Ho et al., 2020).
        """
        with torch.no_grad():
            if t > 1:
                z = torch.randn(x.shape).to(x)
            else:
                z = 0
            e_hat = self.forward(x, t.view(1, 1).repeat(x.shape[0], 1))
            pre_scale = 1 / math.sqrt(self.alpha(t))
            e_scale = (1 - self.alpha(t)) / math.sqrt(1 - self.alpha_bar(t))
            post_sigma = math.sqrt(self.beta(t)) * z
            x = pre_scale * (x - e_scale * e_hat) + post_sigma
            return x

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
