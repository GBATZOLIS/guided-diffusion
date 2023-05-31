import copy
import functools
import os
import time
import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
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

class ScoreVAE(pl.LightningModule):
    def __init__(self, args):
        super(ScoreVAE, self).__init__()

        self.encoder = create_encoder(
        **args_to_dict(args, encoder_defaults().keys()))

        self.diffusion_model, self.diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    def forward(self, x):
        # Define your forward pass.
        x = F.relu(self.layer1(x))
        x = F.softmax(self.layer2(x), dim=1)
        return x

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        # Configure your optimizers and learning rate schedulers
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer