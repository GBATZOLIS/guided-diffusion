import copy
import functools
import os
import time
import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

#new
import pytorch_lightning as pl 
from guided_diffusion.script_util import (
    create_gaussian_diffusion,
    diffusion_defaults,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    encoder_defaults,
    create_encoder
)
from guided_diffusion.resample import create_named_schedule_sampler
from pytorch_lightning.callbacks import Callback
import torch.optim as optim
from torch.optim import AdamW
import torch 
import torchvision
from guided_diffusion.respace import SpacedDiffusion, space_timesteps

class BaseModule(pl.LightningModule):
    def __init__(self, args):
        super(BaseModule, self).__init__()
        self.save_hyperparameters()

        self.args = args

        self.diffusion_model, self.diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        self.diffusion_model = torch.compile(self.diffusion_model)
        
        self.schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, self.diffusion)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x = batch
        t, weights = self.schedule_sampler.sample(x.shape[0], x.device)

        compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.diffusion_model,
                x,
                t
            )

        losses = compute_losses()
        if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

        loss = (losses["loss"] * weights).mean()
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        #self.log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})

        return loss
    
    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x = batch
        t, weights = self.schedule_sampler.sample(x.shape[0], x.device)

        compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.diffusion_model,
                x,
                t
            )

        losses = compute_losses()
        if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

        loss = (losses["loss"] * weights).mean()
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        #self.log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})

        return loss
    
    def sample(self, num_samples=None, time_respacing=""):
        if not num_samples:
            num_samples = self.args.batch_size
        
        diffusion_dict = args_to_dict(self.args, diffusion_defaults().keys())
        diffusion_dict['timestep_respacing'] = time_respacing
        sampling_diffusion = create_gaussian_diffusion(**diffusion_dict)

        sample_fn = (
            sampling_diffusion.p_sample_loop if not self.args.use_ddim else sampling_diffusion.ddim_sample_loop
        )
        
        sample = sample_fn(
            self.diffusion_model,
            (num_samples, 3, self.args.image_size, self.args.image_size),
            clip_denoised=self.args.clip_denoised,
            device=self.device, 
            progress=True
            )

        return sample

    def forward(self, x, timesteps, y=None):
        return self.diffusion_model(x, timesteps, y)

    def configure_optimizers(self):
        class scheduler_lambda_function:
            def __init__(self, warm_up):
                self.use_warm_up = True if warm_up > 0 else False
                self.warm_up = warm_up

            def __call__(self, s):
                if self.use_warm_up:
                    if s < self.warm_up:
                        return s / self.warm_up
                    else:
                        return 1
                else:
                    return 1
        

        optimizer = AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        
        scheduler = {'scheduler': optim.lr_scheduler.LambdaLR(optimizer,scheduler_lambda_function(self.args.lr_anneal_steps)),
                    'interval': 'step'}  # called after each training step
                    
        return [optimizer], [scheduler]

    def log_sample(self, image, name):
        #expected sample range [-1, 1]
        sample = image.detach().cpu()
        sample = ((sample + 1) * 127.5).clamp(0, 255)
        sample = sample.to(torch.float32) / 255.0  # Convert to [0, 1] range
        grid_images = torchvision.utils.make_grid(sample, normalize=False)
        self.logger.experiment.add_image(name, grid_images, self.current_epoch)

    #logging helper
    def log_loss_dict(self, diffusion, ts, losses):
        for key, values in losses.items():
            self.log(key, values.mean())
            # Log the quantiles (four quartiles, in particular).
            for sub_t, sub_loss in zip(ts.cpu().detach().numpy(), values.detach().cpu().numpy()):
                quartile = int(4 * sub_t / diffusion.num_timesteps)
                self.log(f"{key}_q{quartile}", sub_loss, sync_dist=True)

class BaseSampleLoggingCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_validation_epoch_end(self, trainer, pl_module):
        diffusion_samples = pl_module.sample(time_respacing='ddim250')
        pl_module.log_sample(diffusion_samples, name='diffusion_samples')

