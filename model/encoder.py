import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    
    def __init__(self):
        super().__init__(
            #  (Batch_size, Channel=1, Height, width) -> (Batch_size, 128, Height, width)
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            
            # (b_s, 128, h, w) -> (b_s, 128, h, w)
            VAE_ResidualBlock(128, 128),
            
            # (b_s, 128, 32, 32) -> (b_s, 128, 16, 16)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            
            VAE_ResidualBlock(128, 256),
            VAE_ResidualBlock(256, 256),
            
            # (8*8)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            
            VAE_ResidualBlock(256, 512),
            VAE_ResidualBlock(512, 512),
            
            # (4*4)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            
            VAE_AttentionBlock(512),
            
            
            VAE_ResidualBlock(512, 512),
            
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
            
            
        )
        
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (bs, channel=1, h, w)
        # noise: (bs, out_channel, h/8, h/8)
        
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            
            x = module(x)
        
        # (bs, 8, h/8, w/8) -> two tensors of shape (bs, 4, h/8, w/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        
        log_variance = torch.clamp(log_variance, -30, 20)
        
        variance = log_variance.exp()
        
        stdev = variance.sqrt()
        
        # z = N(0, 1) -> N(mean, variance)
        # x = mean + stdev * z
        
        x = mean + stdev * noise
        
        # scale the output by a constant
        x *= 0.18215
        