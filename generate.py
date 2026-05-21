import json
import torch
import random
import argparse
import numpy as np
from scripts import Generator
from scripts.diffusion import PWCDiff
from scripts.diffusion.model import Unet


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str)                 
    parser.set_defaults(ref=True)                      
    args = parser.parse_args()
    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        
if __name__ == '__main__':
    args = parse_args()
    with open(args.config) as fp:
        config = json.load(fp)
        
    set_seed(config['seed'])
    dtype = getattr(torch, config['dtype'])
    
    unet_kwrgs = dict(
        dim=config['model'].get('dim', 64),
        dim_mults=config['model'].get('dim_mults', [1, 2, 4, 8]),
        self_condition=config['model'].get('self_condition', True),
        flash_attn=config['model'].get('flash_attn', True),
        data_type=dtype,
        res_type=config['model'].get('res_type', 'TE')
    )
    unet = Unet(**unet_kwrgs)
    if dtype != torch.float32:
        unet = unet.to(dtype)
    
    pwcdiff_kwargs = dict(
        model=unet,
        image_size=config['diffusion'].get('image_size', 128),
        timesteps=config['diffusion'].get('timesteps', 100),
        sampling_timesteps=config['diffusion'].get('sampling_timesteps', 10),
        ddim_sampling_eta=config['diffusion'].get('ddim_sampling_eta', 0.0),
        objective=config['diffusion'].get('objective', 'pred_x0'),
        beta_schedule=config['diffusion'].get('beta_schedule', 'cosine'),
        use_vgg_loss=config['diffusion'].get('use_vgg_loss', True),
        use_ssim_loss=config['diffusion'].get('use_ssim_loss', True),
        data_type=dtype
    )
    pwcdiff = PWCDiff(**pwcdiff_kwargs)
    
    eval_data_kwargs = dict(
        input_folder=config['eval_data'].get('folder', None),
        mode_name=config['eval_data'].get('mode_name', None),
        sid=config['eval_data'].get('sid', None),
        output_folder=config['eval_data'].get('output_folder', ''),
        data_type=dtype
    )
    
    assert len(eval_data_kwargs['output_folder']) > 0, 'Must specify the predictions folder'

    generator_kwargs = dict(
        eval_batch_size=config['generator'].get('batch_size', 1),
        ckpt=config['generator'].get('milestone', 'best'),
        amp=config['generator'].get('mixed_precision', True),
        illumination_map=config['generator'].get('illumination_map', 'ours'),
        results_folder=config['generator'].get('results_folder', '')
    )
    
    assert len(generator_kwargs['results_folder']) > 0, 'You must results folder to load checkpoints.'
    
    generator = Generator(
        diffusion_model=pwcdiff,
        **eval_data_kwargs,
        **generator_kwargs
    )
    generator.generate()