import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

class ConvNetEncoder(pl.LightningModule):
    def __init__(self, image_size, base_channel_size, latent_dim):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        num_input_channels = 3
        self.latent_dim = latent_dim
        act_fn = nn.GELU

        self.time_conditional = True
        
        out_dim = 2*latent_dim
        c_hid = base_channel_size

        if image_size == 32:
          self.net = nn.Sequential(
              nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
              act_fn(),
              nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
              act_fn(),
              nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
              act_fn(),
              nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
              act_fn(),
              nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
              act_fn(),
              nn.Flatten() # Image grid to single feature vector
          )
          if self.time_conditional:
            self.linear = nn.Linear(2*16*c_hid+1, out_dim)
          else:
            self.linear = nn.Linear(2*16*c_hid, out_dim)
      
        elif image_size == 64:
          self.net = nn.Sequential(
              nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2), # 64x64 => 32x32
              act_fn(),
              nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
              act_fn(),
              nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
              act_fn(),
              nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
              act_fn(),
              nn.Conv2d(2*c_hid, 4*c_hid, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
              act_fn(),
              nn.Conv2d(4*c_hid, 4*c_hid, kernel_size=3, padding=1),
              act_fn(),
              nn.Conv2d(4*c_hid, 4*c_hid, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
              act_fn(),
              nn.Flatten(), # Image grid to single feature vector
            )
          if self.time_conditional:
            self.linear = nn.Linear(4*16*c_hid+1, out_dim)
          else:
            self.linear = nn.Linear(4*16*c_hid, out_dim)

        else:
            raise NotImplementedError('This image size is not supported.')
    
    def forward(self, x, t=None):
        flattened_img = self.net(x)
        if self.time_conditional:
          concat = torch.cat((flattened_img, t[:, None]), dim=1)
          out = self.linear(concat)
        else:
          out = self.linear(flattened_img)

        return out