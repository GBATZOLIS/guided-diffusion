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

class ScoreVAE(pl.LightningModule):
    def __init__(self, args):
        super(ScoreVAE, self).__init__()
        self.save_hyperparameters()

        self.args = args

        self.encoder = create_encoder(
        **args_to_dict(args, encoder_defaults().keys()))

        self.diffusion_model, self.diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        self.schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, self.diffusion)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, cond = batch
        t, weights = self.schedule_sampler.sample(x.shape[0], x.device)

        compute_losses = functools.partial(
                self.diffusion.scoreVAE_training_losses,
                self.encoder,
                self.diffusion_model,
                x,
                t,
                model_kwargs=cond,
                train=True
            )

        losses = compute_losses()
        if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

        loss = (losses["loss"] * weights).mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})

        return loss
    
    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, cond = batch
        t, weights = self.schedule_sampler.sample(x.shape[0], x.device)

        compute_losses = functools.partial(
                self.diffusion.scoreVAE_training_losses,
                self.encoder,
                self.diffusion_model,
                x,
                t,
                model_kwargs=cond,
                train=False
            )

        losses = compute_losses()
        if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

        loss = (losses["loss"] * weights).mean()
        self.log('val_loss', loss, prog_bar=True)
        #self.log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})
        return loss

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
        

        optimizer = AdamW(self.encoder.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        
        scheduler = {'scheduler': optim.lr_scheduler.LambdaLR(optimizer,scheduler_lambda_function(self.args.lr_anneal_steps)),
                    'interval': 'step'}  # called after each training step
                    
        return [optimizer], [scheduler]

    #logging helper
    def log_loss_dict(self, diffusion, ts, losses):
        for key, values in losses.items():
            self.log(key, values.mean())
            # Log the quantiles (four quartiles, in particular).
            for sub_t, sub_loss in zip(ts.cpu().detach().numpy(), values.detach().cpu().numpy()):
                quartile = int(4 * sub_t / diffusion.num_timesteps)
                self.log(f"{key}_q{quartile}", sub_loss)

class LoadAndFreezeModelCallback(Callback):
    def __init__(self, args):
        super().__init__()
        self.diffusion_model_checkpoint = args.diffusion_model_checkpoint

    def on_train_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print(f"Loading pretrained score model from checkpoint: {self.diffusion_model_checkpoint}...")
            pl_module.diffusion_model.load_state_dict(
                th.load(
                    self.diffusion_model_checkpoint, map_location=lambda storage, loc: storage.cuda(pl_module.device)
                )
            )

        # Freeze the unconditional score model
        for param in pl_module.diffusion_model.parameters():
            param.requires_grad = False