import argparse
import random
import time

import torch


# Following parameters come from SinGAN
class get_arguments():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--manualSeed', default = None, type = int)
        parser.add_argument('--not_cuda', action = 'store_true', help = 'disables cuda', default = 0)
        parser.add_argument('--gpu', nargs = '+', default = 0)
        # load, input, save configurations:
        parser.add_argument('--nc_z', type = int, help = 'noise # channels', default = 3)
        parser.add_argument('--nc_im', type = int, help = 'image # channels', default = 3)
        parser.add_argument('--out', help = 'output folder', default = 'Output')

        # networks hyper parameters:
        parser.add_argument('--nfc', type = int, default = 32)
        parser.add_argument('--min_nfc', type = int, default = 32)
        parser.add_argument('--ker_size', type = int, help = 'kernel size', default = 3)
        parser.add_argument('--num_layer', type = int, help = 'number of cnn layers', default = 5)
        parser.add_argument('--stride', help = 'stride', default = 1)
        parser.add_argument('--padd_size', type = int, help = 'net pad size', default = 0)

        # pyramid parameters:
        parser.add_argument('--scale_factor', type = float, help = 'pyramid scale factor',
                            default = 0.75)

        parser.add_argument('--noise_amp', type = float, help = 'addative noise cont weight', default = 0.1)
        parser.add_argument('--min_size', type = int, help = 'image minimal size at the coarser scale', default = 32)
        parser.add_argument('--max_size', type = int, help = 'image maximal size at the coarser scale', default = 256)

        # optimization hyper parameters:
        parser.add_argument('--niter', type = int, default = 2000, help = 'number of epochs to train per scale')
        parser.add_argument('--gamma', type = float, help = 'scheduler gamma', default = 0.1)
        parser.add_argument('--beta1', type = float, default = 0.5, help = 'beta1 for adam. default=0.5')
        parser.add_argument('--Gsteps', type = int, help = 'Generator inner steps', default = 3)
        parser.add_argument('--Dsteps', type = int, help = 'Discriminator inner steps', default = 3)
        parser.add_argument('--lambda_grad', type = float, help = 'gradient penalty weight', default = 10)
        self.parser = parser


# Parameters for Models
class GetReady(get_arguments):

    def __init__(self):
        super(GetReady, self).__init__()
        '''Models'''
        self.parser.add_argument('--pyramid_height', default = 6, type = float,
                                 help = 'pyramid height, max_size * saclr_factor^height '
                                        '= min_size, if pyramid_height set as None,'
                                        'use scale_factor , otherwise using pyramid '
                                        'height to calculate scale_factor')
        self.parser.add_argument('--input_dir', type = str, help = 'input image dir', default = 'Input/Images')
        self.parser.add_argument('--input_name', type = str, default = 'balloons.png', help = 'input image name')
        self.parser.add_argument('--rescale_method', default = 'exsingan', help = '[singan | consingan | exsingan]')
        self.parser.add_argument('--pretrained_path', default = None)
        self.parser.add_argument('--sample_num', default = 36, type = int, help = 'how many samples to generate after '
                                                                                  'training')
        '''Structural GAN'''
        self.parser.add_argument('--use_struct', type = bool, default = True, help = 'whether use struct GAN or not')
        self.parser.add_argument('--struct_batch_size', type = int, default = 16,
                                 help = 'training batch size of structural GAN')
        self.parser.add_argument('--lr_g_struct', default = 1e-4, type = float, help = 'learning rate of generator of '
                                                                                       'structural GAN')
        self.parser.add_argument('--lr_d_struct', default = 1e-4, type = float, help = 'learning rate of discriminator '
                                                                                       'of structural GAN')
        self.parser.add_argument('--struct_epochs', default = 4000, type = int, help = 'training epochs of structural GAN')
        self.parser.add_argument('--D_iters', type = int, default = 2,
                                 help = 'training iters of discriminator of structural GAN')
        self.parser.add_argument('--G_iters', type = int, default = 2,
                                 help = 'training iters of discriminator of structural GAN')
        self.parser.add_argument('--alpha_struct', type = float, default = 10,
                                 help = 'reconstruction loss weight of struct GAN')
        self.parser.add_argument('--struct_channel_dim', default = 64, type = int,
                                 help = 'channel dim of structural GAN')
        self.parser.add_argument('--ref_struct', default = None, type = str,
                                 help = 'whether use pre-trained structural GAN or not')

        '''Semantic GAN'''
        self.parser.add_argument('--perceptual_loss', default = 'vgg', type = str)
        self.parser.add_argument('--use_semantic', type = bool, default = True,
                                 help = 'whether use semantic GAN or not')
        self.parser.add_argument('--use_semantic_layers', nargs = '+', default = ['r53'],
                                 help = 'use which layers of vgg to compute perceptual loss')
        self.parser.add_argument('--semantic_epochs', default = 2000, type = int)
        self.parser.add_argument('--p_weight', default = 0.001, type = float, help = 'the weight of perceptual loss')
        self.parser.add_argument('--semantic_stages', default = [1, 2, 3], type = int, nargs = '+',
                                 help = 'which stages to use semantic GAN')
        self.parser.add_argument('--alpha_semantic', type = float, default = 10,
                                 help = 'reconstruction loss weight of semantic GAN')
        self.parser.add_argument('--lr_g_semantic', type = float, default = 0.0005,
                                 help = 'learning rate of semantic generator')
        self.parser.add_argument('--lr_d_semantic', type = float, default = 0.0005,
                                 help = 'learning rate of semantic discriminator')

        '''Dip GAN'''
        self.parser.add_argument('--use_dip', type = bool, default = True,
                                 help = 'whether use semantic GAN or not')
        self.parser.add_argument('--dip_epochs', default = 5000, type = int)
        self.parser.add_argument('--rec_epoch', default = 0, type = int)
        self.parser.add_argument('--dip_stages', default = [4], type = int, nargs = '+',
                                 help = 'which stages to use semantic GAN')
        self.parser.add_argument('--alpha_dip', type = float, default = 50,
                                 help = 'reconstruction loss weight of semantic GAN')
        self.parser.add_argument('--lr_g_dip', type = float, default = 0.0001,
                                 help = 'learning rate of semantic generator')
        self.parser.add_argument('--lr_d_dip', type = float, default = 0.0001,
                                 help = 'learning rate of semantic discriminator')

        '''Texture GAN'''
        self.parser.add_argument('--use_texture', type = bool, default = True, help = 'whether use texture GAN or not')
        self.parser.add_argument('--alpha_texture', type = float, default = 10,
                                 help = 'reconstruction loss weight of texture GAN')
        self.parser.add_argument('--lr_g_texture', type = float, default = 0.0005,
                                 help = 'learning rate of texture generator')
        self.parser.add_argument('--lr_d_texture', type = float, default = 0.0005,
                                 help = 'learning rate of texture discriminator')
        self.parser.add_argument('--texture_epochs', default = 2000, type = int)
        # self.parser.add_argument('--use_texture_layers', nargs='+', default=['r22', 'r32'],
        #                           help = 'which layers to use for computing texture loss')
        # self.parser.add_argument('--t_weight', default = 1, type = float, help = 'the weight of texture loss')
        # self.parser.add_argument('--l1_weight', default = 1, type = float)

        self.parser.add_argument('--model_name', default = None, type = str, help = 'name of training model')

        '''DGP'''
        self.parser.add_argument('--stds', default = [0.1, 0.2, 0.3], type = float, nargs = '+',
                                 help = 'std of Gaussian distribution to jitter latent coda')
        self.parser.add_argument('--jitter_num', default = 100, type = int, help = 'sampling number of DGP')
        self.parser.add_argument('--cls', default = -1, type = int, help = 'class label of DGP')

        self.parser.add_argument('--use_pe', default = 'SPE', type = str,
                                 choices = ['SPE', 'CSG', None])
        self.parser.add_argument('--pe_dim', default = 8, type = int, help = 'if use_pue == CSG')

        self.parser.add_argument('--mix_ratio', default = 0.3, type = int,
                                 help = '0 to 0.99')
        '''load parameters'''
        opt = self.parser.parse_args()

        opt.device = torch.device("cpu" if opt.not_cuda else "cuda:{}".format(opt.gpu))
        opt.noise_amp_init = opt.noise_amp
        opt.niter_init = opt.niter
        opt.nfc_init = opt.nfc
        opt.min_nfc_init = opt.min_nfc
        opt.scale_factor_init = opt.scale_factor
        opt.timestamp = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))

        if opt.manualSeed is None:
            opt.manualSeed = 0
        # torch.backends.cudnn.deterministic = True
        random.seed(opt.manualSeed)
        torch.manual_seed(opt.manualSeed)

        if torch.cuda.is_available() and opt.not_cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

        self.opt = opt
