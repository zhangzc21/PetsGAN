import argparse


def add_args():
    parser = argparse.ArgumentParser()

    ### Pipeline stuff ###
    parser.add_argument(
        '--eval_mode', action='store_true', default=False,
        help='Evaluation mode? (do not save logs) '
             ' (default: %(default)s)')
    ### Model stuff ###
    parser.add_argument(
        '--model', type=str, default='BigGAN',
        help='Name of the model module (default: %(default)s)')
    parser.add_argument(
        '--G_param', type=str, default='SN',
        help='Parameterization style to use for G, spectral norm (SN) or SVD (SVD)'
             ' or None (default: %(default)s)')
    parser.add_argument(
        '--D_param', type=str, default='SN',
        help='Parameterization style to use for D_lr, spectral norm (SN) or SVD (SVD)'
             ' or None (default: %(default)s)')
    parser.add_argument(
        '--G_ch', type=int, default=96,
        help='Channel multiplier for G (default: %(default)s)')
    parser.add_argument(
        '--D_ch', type=int, default=96,
        help='Channel multiplier for D_lr (default: %(default)s)')
    parser.add_argument(
        '--G_depth', type=int, default=1,
        help='Number of resblocks per stage in G? (default: %(default)s)')
    parser.add_argument(
        '--D_depth', type=int, default=1,
        help='Number of resblocks per stage in D_lr? (default: %(default)s)')
    parser.add_argument(
        '--D_thin', action='store_false', dest='D_wide', default=True,
        help='Use the SN-GAN channel pattern for D_lr? (default: %(default)s)')
    parser.add_argument(
        '--G_shared', default=True,
        help='Use shared embeddings in G? (default: %(default)s)')
    parser.add_argument(
        '--shared_dim', type=int, default=128,
        help='G''s shared embedding dimensionality; if 0, will be equal to dim_z. '
             '(default: %(default)s)')
    parser.add_argument(
        '--dim_z', type=int, default=120,
        help='Noise dimensionality: %(default)s)')
    parser.add_argument(
        '--hier', default=True,
        help='Use hierarchical z in G? (default: %(default)s)')
    parser.add_argument(
        '--n_classes', type=int, default=1000,
        help='Number of class conditions %(default)s)')
    parser.add_argument(
        '--cross_replica', action='store_true', default=False,
        help='Cross_replica batchnorm in G?(default: %(default)s)')
    parser.add_argument(
        '--mybn', action='store_true', default=False,
        help='Use my batchnorm (which supports standing stats?) %(default)s)')
    parser.add_argument(
        '--G_nl', type=str, default='inplace_relu',
        help='Activation function for G (default: %(default)s)')
    parser.add_argument(
        '--D_nl', type=str, default='inplace_relu',
        help='Activation function for D_lr (default: %(default)s)')
    parser.add_argument(
        '--G_attn', type=str, default='64',
        help='What resolutions to use attention on for G (underscore separated) '
             '(default: %(default)s)')
    parser.add_argument(
        '--D_attn', type=str, default='64',
        help='What resolutions to use attention on for D_lr (underscore separated) '
             '(default: %(default)s)')
    parser.add_argument(
        '--norm_style', type=str, default='bn',
        help='Normalizer style for G, one of bn [batchnorm], in [instancenorm], '
             'ln [layernorm], gn [groupnorm] (default: %(default)s)')

    ### Model init stuff ###
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed to use; affects both initialization and '
             ' dataloading. (default: %(default)s)')
    parser.add_argument(
        '--G_init', type=str, default='ortho',
        help='Init style to use for G (default: %(default)s)')
    parser.add_argument(
        '--D_init', type=str, default='ortho',
        help='Init style to use for D_lr(default: %(default)s)')
    parser.add_argument(
        '--skip_init', default=True,
        help='Skip initialization, ideal for testing when ortho init was used '
             '(default: %(default)s)')

    ### Optimizer stuff ###
    parser.add_argument(
        '--optimizer', type=str, default='Adam',
        help='Optimizer, Adam or SGD (default: %(default)s)')
    parser.add_argument(
        '--G_lr', type=float, default=5e-5,
        help='Learning rate to use for Generator (default: %(default)s)')
    parser.add_argument(
        '--D_lr', type=float, default=2e-4,
        help='Learning rate to use for Discriminator (default: %(default)s)')
    parser.add_argument(
        '--Z_lr_mult', type=float, default=50,
        help='Learning rate multiplication to use for Z (default: %(default)s)')
    parser.add_argument(
        '--G_B1', type=float, default=0.0,
        help='Beta1 to use for Generator (default: %(default)s)')
    parser.add_argument(
        '--D_B1', type=float, default=0.0,
        help='Beta1 to use for Discriminator (default: %(default)s)')
    parser.add_argument(
        '--G_B2', type=float, default=0.999,
        help='Beta2 to use for Generator (default: %(default)s)')
    parser.add_argument(
        '--D_B2', type=float, default=0.999,
        help='Beta2 to use for Discriminator (default: %(default)s)')

    ### Batch size, parallel, and precision stuff ###
    parser.add_argument(
        '--G_fp16', action='store_true', default=False,
        help='Train with half-precision in G? (default: %(default)s)')
    parser.add_argument(
        '--D_fp16', action='store_true', default=False,
        help='Train with half-precision in D_lr? (default: %(default)s)')
    parser.add_argument(
        '--D_mixed_precision', action='store_true', default=False,
        help='Train with half-precision activations but fp32 params in D_lr? '
             '(default: %(default)s)')
    parser.add_argument(
        '--G_mixed_precision', action='store_true', default=False,
        help='Train with half-precision activations but fp32 params in G? '
             '(default: %(default)s)')
    parser.add_argument(
        '--accumulate_stats', action='store_true', default=False,
        help='Accumulate "standing" batchnorm stats? (default: %(default)s)')
    parser.add_argument(
        '--num_standing_accumulations', type=int, default=16,
        help='Number of forward passes to use in accumulating standing stats? '
             '(default: %(default)s)')

    ### Bookkeping stuff ###
    parser.add_argument(
        '--weights_root', type=str, default='Pretrained',
        help='Default location to store weights (default: %(default)s)')

    ### G_EMA Stuff ###
    parser.add_argument(
        '--use_ema', default=True,
        help='Use the G_EMA parameters of G for evaluation? (default: %(default)s)')

    ### Numerical precision and SV stuff ###
    parser.add_argument(
        '--adam_eps', type=float, default=1e-6,
        help='epsilon value to use for Adam (default: %(default)s)')
    parser.add_argument(
        '--BN_eps', type=float, default=1e-5,
        help='epsilon value to use for BatchNorm (default: %(default)s)')
    parser.add_argument(
        '--SN_eps', type=float, default=1e-6,
        help='epsilon value to use for Spectral Norm(default: %(default)s)')
    parser.add_argument(
        '--num_G_SVs', type=int, default=1,
        help='Number of SVs to track in G (default: %(default)s)')
    parser.add_argument(
        '--num_D_SVs', type=int, default=1,
        help='Number of SVs to track in D_lr (default: %(default)s)')
    parser.add_argument(
        '--num_G_SV_itrs', type=int, default=1,
        help='Number of SV itrs in G (default: %(default)s)')
    parser.add_argument(
        '--num_D_SV_itrs', type=int, default=1,
        help='Number of SV itrs in D_lr (default: %(default)s)')

    ### Resume training stuff
    parser.add_argument(
        '--load_weights', type=str, default='256',
        help='Suffix for which weights to load (e.g. best0, copy0) '
             '(default: %(default)s)')

    ### Log stuff ###
    parser.add_argument(
        '--no_tb', action='store_true', default=False,
        help='Do not use tensorboard? '
             '(default: %(default)s)')
    ##########################################################################
    parser.add_argument(
        '--image_path', type=str, default='data/ILSVRC2012_val_00000525.JPEG',
        help='Path of the image to be processed (default: %(default)s)')
    parser.add_argument(
        '--ftr_type', type=str, default='Discriminator',
        choices=['Discriminator', 'VGG'],
        help='Feature loss type, choose from Discriminator and VGG (default: %(default)s)')
    parser.add_argument(
        '--random_G', action='store_true', default=False,
        help='Use randomly initialized generator? (default: %(default)s)')
    parser.add_argument('--gan_mode', default='biggan',
                        help='GAT with sparse version or not.')
    parser.add_argument(
        '--resolution', type=int, default=256,
        help='Resolution to resize the input image (default: %(default)s)')
    parser.add_argument(
        '--ftr_num', type=int, default=[8, 8, 8, 8, 8], nargs='+',
        help='Number of features to computer feature loss (default: %(default)s)')
    parser.add_argument(
        '--class', type=int, default=-1,
        help='class index of the image (default: %(default)s)')

    parser.add_argument(
        '--iterations', type=int, default=[200, 200, 200, 200, 200], nargs='+',
        help='Training iterations for all stages')

    parser.add_argument(
        '--gpu_ids', type=int, default=[0], nargs='+',
        help='which gpus to use')

    parser.add_argument(
        '--lambda_D', type=float, default=1,
        help='Discriminator feature loss weight (default: %(default)s)')
    parser.add_argument(
        '--lambda_LK', type=float, default=0.02,
        help='Weight for the negative log-likelihood loss (default: %(default)s)')
    parser.add_argument(
        '--lambda_MSE', type=float, default=1,
        help='MSE loss weight (default: %(default)s)')
    parser.add_argument(
        '--lambda_P', type=float, default=0,
        help='Prior loss weight (default: %(default)s)')
    parser.add_argument(
        '--ftpg', type=int, default=[2, 3, 4, 5, 7], nargs='+',
        help='Number of parameter groups to finetune (default: %(default)s)')
    parser.add_argument(
        '--G_lrs', type=float, default=[5e-5] * 5, nargs='+',
        help='Learning rate to use for Generator (default: %(default)s)')
    parser.add_argument(
        '--z_lrs', type=float, default=[2e-3, 2e-3, 2e-3, 2e-4, 2e-5], nargs='+',
        help='Learning rate to use for Discriminator (default: %(default)s)')
    parser.add_argument(
        '--use_Dscore', type=bool, default=True,
        help='use D_lr output not(default: %(default)s)')

    return parser
