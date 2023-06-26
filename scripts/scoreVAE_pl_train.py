"""
Train a diffusion model on images.
"""

import argparse
import hydra
from hydra import compose, initialize

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
from guided_diffusion.scoreVAE_pl_module import ScoreVAE, ScoreVAESampleLoggingCallback, ScoreVAETestSetupCallback
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torch
import psutil
import os

def print_memory_usage():
    mem_info = psutil.virtual_memory()
    print(f'Used memory: {mem_info.used / 1024**3:.2f} GB')
    print(f'Total memory: {mem_info.total / 1024**3:.2f} GB')
    print(f'Free memory: {mem_info.free / 1024**3:.2f} GB')
    print(f'Memory percentage used: {mem_info.percent}%')

def create_datamodule(args):
    if args.dataset == 'cifar10':
        datamodule = Cifar10DataModule(args)
    else:
        datamodule = ImageDataModule(data_dir=args.data_dir,
                                     dataset=args.dataset,
                                     percentage_use = 100,
                                     batch_size=args.batch_size,
                                     image_size=args.image_size,
                                     class_cond=args.class_cond,
                                     num_workers=args.workers)
    return datamodule



#@hydra.main(config_path='../configs', config_name="train")
def main(config):
    #args = create_argparser().parse_args()
    args = config.args

    datamodule = create_datamodule(args)

    if args.phase == 'train':
        logger = pl.loggers.TensorBoardLogger(args.log_dir, name='', version=args.log_name)
        callbacks = [EMA(decay=float(args.ema_rate)), ScoreVAESampleLoggingCallback()]
        trainer = pl.Trainer(accelerator = args.accelerator,
                            #strategy='ddp_find_unused_parameters_true',
                            devices=args.gpus,
                            num_nodes = args.num_nodes,
                            accumulate_grad_batches = args.accumulate_grad_batches,
                            gradient_clip_val = args.grad_clip,
                            max_steps=args.step_limit, 
                            max_epochs =args.epochs_limit,
                            callbacks=callbacks, 
                            logger = logger,
                            num_sanity_val_steps=0
                            )

        compiled_model = ScoreVAE(args)
        #compiled_model = torch.compile(ScoreVAE(args))
        # Set the precision for 32-bit floating point matrix multiplication
        #torch.set_float32_matmul_precision('medium')  # or 'high'
        trainer.fit(compiled_model, datamodule=datamodule, ckpt_path=args.resume_checkpoint)
    
    elif args.phase == 'test':
        logger = pl.loggers.TensorBoardLogger(os.path.join(args.log_dir, 'test'), name='', version=args.log_name)
        callbacks = [ScoreVAETestSetupCallback()]
        trainer = pl.Trainer(accelerator = args.accelerator,
                            #strategy='ddp_find_unused_parameters_true',
                            devices=args.gpus,
                            num_nodes = args.num_nodes,
                            accumulate_grad_batches = args.accumulate_grad_batches,
                            gradient_clip_val = args.grad_clip,
                            max_steps=args.step_limit, 
                            max_epochs=args.epochs_limit,
                            callbacks=callbacks, 
                            logger = logger,
                            num_sanity_val_steps=1,
                            inference_mode=False
                            )

        pl_model = ScoreVAE(args)
        trainer.test(pl_model, datamodule=datamodule, ckpt_path=args.resume_checkpoint)
    
    elif args.phase == 'inspection':
        pl_module = ScoreVAE(args)
        pl_module = pl_module.load_from_checkpoint(args.resume_checkpoint, args=args)

        datamodule.setup()
        dataloader = datamodule.val_dataloader()
        batch = next(iter(dataloader))
        x, cond = pl_module._handle_batch(batch)
        pl_module.inspect_encoder_profile(x.to(pl_module.device))



def create_argparser():
    defaults = dict(
        phase='train',
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
        accelerator = 'gpu', #new
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
        
        ### ScoreVAE settings #new
        diffusion_model_checkpoint="",
        scoreVAE_training_loss='vlb',
        beta=0.01,
        latent_dim = 1024, 
        encoder_use_fp16 = False,
        encoder_width = 128,
        encoder_depth = 2,
        encoder_attention_resolutions = "32,16,8",
        encoder_use_scale_shift_norm = True,
        encoder_resblock_updown = True,
        encoder_pool = "attention",
        encoder_type = 'HalfUnet',

        #sampling settings
        #CLIP_DENOISED IS AN IMPORTANT SETTING. IT MUST BE SET TO TRUE.
        clip_denoised=True, #THIS MUST BE TRUE IF YOU USE PSAMPLE - OTHERWISE THE SAMPLES ARE OUT OF DISTRIBUTION
        use_ddim=False
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    return parser


if __name__ == "__main__":
    # Create a parser
    parser = argparse.ArgumentParser(description='Provide configuration path and name.')

    # Add arguments
    parser.add_argument('--config_path', type=str, default='../configs', help='Relative path to configuration files.')
    parser.add_argument('--config_name', type=str, default='train', help='Name of the configuration file.')

    # Parse the arguments
    args = parser.parse_args()

    # Use the config path and name provided in command line arguments
    initialize(version_base=None, config_path=args.config_path, job_name="test_app")

    home_path = os.path.expanduser('~')
    cfg = compose(config_name=args.config_name, overrides=[f"args.home={home_path}"])

    # Run your main function here
    main(cfg)
