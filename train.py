import os
import json
import torch
import random
import argparse
import numpy as np
from scripts import Trainer
from termcolor import colored
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
        res_type=config['model'].get('res_type', 'TE')
    )
    unet = Unet(**unet_kwrgs)
    if dtype != torch.float32:
        unet = unet.to(dtype)
    
    pwcdiff_kwargs = dict(
        model=unet,
        image_size=config['diffusion'].get('image_size', 128),
        timesteps=config['diffusion'].get('timesteps', 100),
        sampling_timesteps=config['diffusion'].get('sampling_timesteps', 25),
        objective=config['diffusion'].get('objective', 'pred_x0'),
        beta_schedule=config['diffusion'].get('beta_schedule', 'cosine'),
        use_vgg_loss=config['diffusion'].get('use_vgg_loss', True),
        use_ssim_loss=config['diffusion'].get('use_ssim_loss', True),
        data_type=dtype
    )
    pwcdiff = PWCDiff(**pwcdiff_kwargs)
    
    pytorch_total_params = sum(p.numel() for p in pwcdiff.parameters())
    pytorch_total_train_params = sum(p.numel() for p in pwcdiff.parameters() if p.requires_grad)
    
    print(colored(f'Total Params: {pytorch_total_params}', 'red'))
    print(colored(f'Total Trainable Params: {pytorch_total_train_params}', 'red'))
    
    train_data_kwargs = dict(
        train_folders=config['train_data'].get('folders', None),
        train_mode_names=config['train_data'].get('mode_names', None),
        train_sid=config['train_data'].get('sid', None),
        data_type=dtype
    )
    
    eval_data_kwargs = dict(
        eval_folders=config['eval_data'].get('folders', None),
        eval_mode_names=config['eval_data'].get('mode_names', None),
        eval_sid=config['eval_data'].get('sid', None),
    )
    
    trainer_kwargs = dict(
        train_batch_size=config['trainer'].get('batch_size', 16),
        train_lr=config['trainer'].get('train_lr', 8e-5),
        train_num_steps=config['trainer'].get('train_steps', 1000000),
        gradient_accumulate_every=config['trainer'].get('gradient_accumulate', 2),
        ema_decay=config['trainer'].get('ema_decay', 0.995),
        amp=config['trainer'].get('mixed_precision', True),
        calculate_psnr=config['trainer'].get('calculate_psnr', True),
        save_and_sample_every=config['trainer'].get('save_steps', 1000),
        save_best_and_latest_only=config['trainer'].get('save_best_and_latest_only', True),
        illumination_map=config['trainer'].get('illumination_map', 'ours'),
        results_folder=config['trainer'].get('results_folder', ''),
        resume=config['trainer'].get('resume', False),
        job_name=config['trainer'].get('job_name', 'Train PWC-Diff')
    )
    
    assert len(trainer_kwargs['results_folder']) > 0, 'You must results folder to save checkpoints.'
    
    trainer = Trainer(
        diffusion_model=pwcdiff,
        **train_data_kwargs,
        **eval_data_kwargs,
        **trainer_kwargs
    )
    trainer.train()
