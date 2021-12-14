# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 20:21:35 2020

@author: Administrator
"""
import torch

from BigGAN.inversion import Inversion
from Utils.utils import seed_rng


def RunJitter(image_path, stds = [0.1, 0.2, 0.3, 0.4, 0.5], num_sample = 100, cls = -1):

    config = {'eval_mode': False, 'model': 'BigGAN', 'G_param': 'SN', 'D_param': 'SN', 'G_ch': 96, 'D_ch': 96,
              'G_depth': 1, 'D_depth': 1, 'D_wide': True, 'G_shared': True, 'shared_dim': 128, 'dim_z': 120,
              'hier': True, 'n_classes': 1000, 'cross_replica': False, 'mybn': False, 'G_nl': 'inplace_relu',
              'D_nl': 'inplace_relu', 'G_attn': '64', 'D_attn': '64', 'norm_style': 'bn', 'seed': 0, 'G_init': 'ortho',
              'D_init': 'ortho', 'skip_init': True, 'optimizer': 'Adam', 'G_lr': 5e-05, 'D_lr': 0.0002, 'Z_lr_mult': 50,
              'G_B1': 0.0, 'D_B1': 0.0, 'G_B2': 0.999, 'D_B2': 0.999, 'G_fp16': False, 'D_fp16': False,
              'D_mixed_precision': False, 'G_mixed_precision': False, 'accumulate_stats': False,
              'num_standing_accumulations': 16, 'weights_root': 'Pretrained', 'use_ema': True, 'adam_eps': 1e-06,
              'BN_eps': 1e-05, 'SN_eps': 1e-06, 'num_G_SVs': 1, 'num_D_SVs': 1, 'num_G_SV_itrs': 1, 'num_D_SV_itrs': 1,
              'load_weights': '128', 'no_tb': False, 'image_path': image_path, 'ftr_type': 'Discriminator',
              'random_G': False, 'gan_mode': 'biggan', 'resolution': 128, 'ftr_num': [5, 5, 5], 'class': cls,
              'iterations': [125, 125, 100], 'gpu_ids': [0], 'lambda_D': 1, 'lambda_LK': 0.02, 'lambda_MSE': 1,
              'lambda_P': 0, 'ftpg': [7, 7, 7], 'G_lrs': [2e-07, 2e-05, 2e-06], 'z_lrs': [0.1, 0.01, 0.0002],
              'use_Dscore': True}

    seed_rng(0)
    IG = Inversion(config)
    IG.set_generator()
    IG.set_discriminator()
    IG.set_image()
    IG.select_z(num_z = 500)
    IG.set_optimizer()
    IG.inversion()
    IG.jitter(save_path = IG.path_, stds = stds, jitter_num = num_sample)
    torch.cuda.empty_cache()
