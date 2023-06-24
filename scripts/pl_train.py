"""
Train a diffusion model on images.
"""

import argparse

from guided_diffusion.pl_image_datasets import ImageDataModule, Cifar10DataModule
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    encoder_defaults,
    create_encoder
)

from guided_diffusion.pl_ema import EMA
from guided_diffusion.pl_module import BaseModule, BaseSampleLoggingCallback
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torch
import psutil
from pathlib import Path

def print_memory_usage():
    mem_info = psutil.virtual_memory()
    print(f'Used memory: {mem_info.used / 1024**3:.2f} GB')
    print(f'Total memory: {mem_info.total / 1024**3:.2f} GB')
    print(f'Free memory: {mem_info.free / 1024**3:.2f} GB')
    print(f'Memory percentage used: {mem_info.percent}%')

def main():
    args = create_argparser().parse_args()
    
    if args.dataset == 'cifar10':
        datamodule = Cifar10DataModule(args)
    else:
        datamodule = ImageDataModule(data_dir=args.data_dir,
                                     dataset=args.dataset,
                                     batch_size=args.batch_size,
                                     image_size=args.image_size,
                                     class_cond=args.class_cond,
                                     num_workers=args.workers)

    print("training...")
    print(args.log_dir)
    log_dir = Path(args.log_dir)
    print(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = pl.loggers.TensorBoardLogger(save_dir=log_dir, name='', version=args.log_name)

    callbacks = [EMA(decay=float(args.ema_rate)), BaseSampleLoggingCallback()]
    trainer = pl.Trainer( accelerator = 'gpu' if args.gpus > 0 else 'cpu',
                          #strategy='ddp_find_unused_parameters_true',
                          devices=1 if args.gpus == 0 else args.gpus,
                          num_nodes = args.num_nodes,
                          accumulate_grad_batches = args.accumulate_grad_batches,
                          gradient_clip_val = args.grad_clip,
                          max_steps=args.step_limit, 
                          max_epochs =args.epochs_limit,
                          callbacks=callbacks, 
                          logger = logger,
                          num_sanity_val_steps=0
                          )

    model = BaseModule(args)

    # Set the precision for 32-bit floating point matrix multiplication
    #torch.set_float32_matmul_precision('medium')  # or 'high'
    
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume_checkpoint)

def create_argparser():
    defaults = dict(
        data_dir="",
        dataset="",
        log_dir="", 
        log_name="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        gpus=1, #new
        num_nodes=1, #new
        accelerator = None, #new
        workers = 4, #new
        accumulate_grad_batches=1, #new
        grad_clip = 0., #new 
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint=None,
        step_limit = 2000000, #new
        epochs_limit = 10000, #new
        use_fp16=False,
        fp16_scale_growth=1e-3,

        #sampling settings
        clip_denoised=True,
        use_ddim=True
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    
    add_dict_to_argparser(parser, defaults)

    return parser


if __name__ == "__main__":
    main()
