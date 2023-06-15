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
import lpips

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

        # store the diffusion model checkpoint path
        self.diffusion_model_checkpoint = args.diffusion_model_checkpoint
    
    def on_train_start(self):
        # Load the pretrained diffusion model
        if self.trainer.global_rank == 0:
            print(f"Loading pretrained score model from checkpoint: {self.diffusion_model_checkpoint}...")
            
            # load the whole checkpoint
            checkpoint = torch.load(self.diffusion_model_checkpoint, map_location=self.device)
            
            # Create a new state_dict with corrected key names if necessary
            if any(k.startswith("diffusion_model.") for k in checkpoint['state_dict'].keys()):
                corrected_state_dict = {k.replace("diffusion_model.", ""): v for k, v in checkpoint['state_dict'].items()}
            else:
                corrected_state_dict = checkpoint['state_dict']

            # Load only the diffusion_model parameters
            self.diffusion_model.load_state_dict(corrected_state_dict)

        # Freeze the diffusion model
        for param in self.diffusion_model.parameters():
            param.requires_grad = False

        # Convert the diffusion model to FP16
        #self.diffusion_model.convert_to_fp16()
        #self.diffusion_model.dtype = torch.float16

        
    def _handle_batch(self, batch):
        if type(batch) == list:
            x, cond = batch
        else:
            x, cond = batch, {}
        return x, cond

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, cond = self._handle_batch(batch)
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
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        self.log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})

        return loss
    
    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, cond = self._handle_batch(batch)
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
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        #self.log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})


        return loss
    
    def test_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, cond = self._handle_batch(batch)
        z = self.encode(x)
        reconstructed_samples = self.reconstruct(z, time_respacing='250')
        avg_lpips_score = torch.mean(self.lpips_distance_fn(reconstructed_samples.to(self.device), x.to(self.device)))

        self.log_sample(x, name='input_samples')
        self.log_sample(reconstructed_samples, name='reconstructed_samples')

        difference = torch.flatten(reconstructed_samples, start_dim=1) - torch.flatten(x, start_dim=1)
        L2norm = torch.linalg.vector_norm(difference, ord=2, dim=1)
        avg_L2norm = torch.mean(L2norm)

        # Logging LPIPS score
        self.log('LPIPS', avg_lpips_score, on_step=True, on_epoch=True, prog_bar=True)

        # Logging L2 norm
        self.log('L2', avg_L2norm, on_step=True, on_epoch=True, prog_bar=True)
    
        # Return metrics
        return {'LPIPS': avg_lpips_score, 'L2': avg_L2norm}

    def encode(self, x):
        #compute the parameters of the encoding distribution p_φ(z|x_t)
        latent_distribution_parameters = self.encoder(x, th.zeros(size=(x.size(0),)).to(self.device))
        latent_dim = latent_distribution_parameters.size(1)//2
        mean_z = latent_distribution_parameters[:, :latent_dim]
        log_var_z = latent_distribution_parameters[:, latent_dim:]

        #sample the latent factor z
        z = mean_z + th.sqrt(log_var_z.exp())*th.randn_like(mean_z)
        return z
    
    def reconstruct(self, z, time_respacing=""):
        def get_encoder_correction_fn(encoder):
            def get_log_density_fn(encoder):
                def log_density_fn(x, z, t):
                    latent_distribution_parameters = encoder(x, t)
                    latent_dim = latent_distribution_parameters.size(1)//2
                    mean_z = latent_distribution_parameters[:, :latent_dim]
                    log_var_z = latent_distribution_parameters[:, latent_dim:]
                    logdensity = -1/2*th.sum(th.square(z - mean_z)/log_var_z.exp(), dim=1)
                    return logdensity
                    
                return log_density_fn

            def encoder_correction_fn(x_t, t, z):
                th.set_grad_enabled(True)
                x = x_t.detach()
                x.requires_grad_()
                #t = t.float().requires_grad_(True)  #new 
                #z = z.requires_grad_(True)  #new 
                log_density_fn = get_log_density_fn(encoder)
                device = x.device
                ftx = log_density_fn(x, z, t)
                # Check requires_grad and grad_fn
                #for var_name, tensor in [('x', x), ('t', t), ('z', z), ('ftx', ftx)]:
                #    print(f"{var_name} requires_grad: {tensor.requires_grad}, grad_fn: {tensor.grad_fn}")
                
                grad_log_density = th.autograd.grad(outputs=ftx, inputs=x,
                                      grad_outputs=th.ones(ftx.size()).to(device),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
                th.set_grad_enabled(False)
                return grad_log_density

            return encoder_correction_fn
        
        encoder_correction_fn = get_encoder_correction_fn(self.encoder)
        cond_kwargs={}
        cond_kwargs['z'] = z

        sampling_diffusion = create_gaussian_diffusion(
                                    steps=self.args.diffusion_steps,
                                    learn_sigma=self.args.learn_sigma,
                                    noise_schedule=self.args.noise_schedule,
                                    use_kl=self.args.use_kl,
                                    predict_xstart=self.args.predict_xstart,
                                    rescale_timesteps=self.args.rescale_timesteps,
                                    rescale_learned_sigmas=self.args.rescale_learned_sigmas,
                                    timestep_respacing=time_respacing)
        '''
        sample_fn = (
            sampling_diffusion.p_sample_loop if not self.args.use_ddim else sampling_diffusion.ddim_sample_loop
        )
        '''
        sample_fn = sampling_diffusion.ddim_sample_loop
        
        sample = sample_fn(
            self.diffusion_model,
            (z.size(0), 3, self.args.image_size, self.args.image_size),
            clip_denoised=self.args.clip_denoised,
            cond_fn=encoder_correction_fn,
            cond_kwargs=cond_kwargs,
            device=self.device,
            progress=True
        )

        return sample #expected range [-1, 1] for images (depends on the preprocessed values)

    def sample_from_diffusion_model(self, num_samples=None, time_respacing=""):
        if not num_samples:
            num_samples = self.args.batch_size
        
        sampling_diffusion = create_gaussian_diffusion(
                                    steps=self.args.diffusion_steps,
                                    learn_sigma=self.args.learn_sigma,
                                    noise_schedule=self.args.noise_schedule,
                                    use_kl=self.args.use_kl,
                                    predict_xstart=self.args.predict_xstart,
                                    rescale_timesteps=self.args.rescale_timesteps,
                                    rescale_learned_sigmas=self.args.rescale_learned_sigmas,
                                    timestep_respacing=time_respacing)

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

'''
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
'''

class ScoreVAETestSetupCallback(Callback):
    def __init__(self):
        super().__init__()
        self.lpips_distance_fn = lpips.LPIPS(net='vgg')

    def setup(self, trainer, pl_module, stage):
        self.lpips_distance_fn = self.lpips_distance_fn.to(pl_module.device)

class ScoreVAESampleLoggingCallback(Callback):
    def __init__(self):
        super().__init__()
        self.lpips_distance_fn = lpips.LPIPS(net='vgg')

    def setup(self, trainer, pl_module, stage):
        self.lpips_distance_fn = self.lpips_distance_fn.to(pl_module.device)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == 2:
            diffusion_samples = pl_module.sample_from_diffusion_model(time_respacing='ddim250')
            pl_module.log_sample(diffusion_samples, name='diffusion_samples')

        if trainer.current_epoch == 0 or (trainer.current_epoch+1) % 5 ==0:
            
            # Obtain a batch from the validation dataloader
            dataloader = trainer.datamodule.val_dataloader()
            batch = next(iter(dataloader))
            x, cond = pl_module._handle_batch(batch)

            # Generate sample using the encode and reconstruct methods
            input_samples = x.to(pl_module.device)
            z = pl_module.encode(input_samples)
            reconstructed_samples = pl_module.reconstruct(z, time_respacing='ddim250')

            self.lpips_distance_fn = self.lpips_distance_fn.to(pl_module.device)
            avg_lpips_score = torch.mean(self.lpips_distance_fn(reconstructed_samples.to(pl_module.device), input_samples.to(pl_module.device)))
            #avg_lpips_score = trainer.training_type_plugin.reduce(avg_lpips_score, reduction='mean')

            pl_module.log('LPIPS', avg_lpips_score.detach(), on_step=False, on_epoch=True, prog_bar=False, logger=True)

            # Log the generated samples
            difference = (reconstructed_samples - input_samples).detach().cpu()
            difference_grid = torchvision.utils.make_grid(difference, normalize=True, scale_each=True)
            pl_module.logger.experiment.add_image('difference', difference_grid, pl_module.current_epoch)
            
            pl_module.log_sample(input_samples, name='input_samples')
            pl_module.log_sample(reconstructed_samples, name='reconstructed_samples')

            difference = torch.flatten(reconstructed_samples, start_dim=1) - torch.flatten(input_samples, start_dim=1)
            L2norm = torch.linalg.vector_norm(difference, ord=2, dim=1)
            avg_L2norm = torch.mean(L2norm)
            #avg_L2norm = trainer.training_type_plugin.reduce(avg_L2norm, reduction='mean')

            # Log the average L2 norm
            pl_module.log('L2', avg_L2norm.detach(), on_step=False, on_epoch=True, prog_bar=False, logger=True)



