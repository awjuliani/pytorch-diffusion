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
        ### Generates a gif containing a grid of samples from the model
        gif_shape = [3, 3]
        n_hold_final = 10
        sample_batch_size = gif_shape[0] * gif_shape[1]
        gen_samples = pl_module.diffuse_sample(sample_batch_size, n_hold_final)
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
