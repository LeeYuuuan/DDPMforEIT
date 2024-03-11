import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 32
HEIGHT = 32
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(voltages, conductivity,strength=0.8, do_cfg=False, cfg_scale=7.5, sampler_name="ddpm", n_inference_steps=50, models={}, seed=None,
                device=None, idle_device=None):
    
    with torch.no_grad():
        
        if not(0 < strength <=1):
            raise ValueError("strength must be between 0 and 1")
        
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x
            
        # Initialize random number generator according to the seed specified
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)
        
        
