import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
import imageio
import os


class SampleImages(Callback):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.depth = dataset.depth
        self.size = dataset.size

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        # generate images
        with torch.no_grad():
            gif_shape = [5, 5]
            sample_batch_size = gif_shape[0] * gif_shape[1]
            n_hold_final = 10

            # Generate samples from denoising process
            gen_samples = []
            x = torch.randn((sample_batch_size, self.depth, self.size, self.size)).to(
                pl_module.device
            )
            sample_steps = torch.arange(pl_module.t_range - 1, 0, -1).to(
                pl_module.device
            )
            for t in sample_steps:
                x = pl_module.denoise_sample(x, t)
                if t % 50 == 0:
                    gen_samples.append(x)
            for _ in range(n_hold_final):
                gen_samples.append(x)
            gen_samples = torch.stack(gen_samples, dim=0).moveaxis(2, 4).squeeze(-1)
            gen_samples = (gen_samples.clamp(-1, 1) + 1) / 2

            gen_samples = (gen_samples * 255).type(torch.uint8)
            gen_samples = gen_samples.reshape(
                -1,
                gif_shape[0],
                gif_shape[1],
                self.size,
                self.size,
                self.depth,
            )

            def stack_samples(gen_samples, stack_dim):
                gen_samples = list(torch.split(gen_samples, 1, dim=1))
                for i in range(len(gen_samples)):
                    gen_samples[i] = gen_samples[i].squeeze(1)
                return torch.cat(gen_samples, dim=stack_dim)

            gen_samples = stack_samples(gen_samples, 2)
            gen_samples = stack_samples(gen_samples, 2).cpu().numpy()

            if not os.path.exists(f"{trainer.logger.log_dir}"):
                os.makedirs(f"{trainer.logger.log_dir}")
            imageio.mimsave(
                f"{trainer.logger.log_dir}/pred_{trainer.global_step}.gif",
                list(gen_samples),
                fps=5,
            )
