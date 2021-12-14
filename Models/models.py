import torch
import torch.nn as nn
from Models.blocks import ResBlocks, ResBlock, ConvBlocks, ConvBlock
from Utils import PosEncoding
import functools
# from halonet_pytorch import HaloAttention
import torch.nn.functional as F
from Utils import functions
import einops

class upsample(nn.Module):
    def __init__(self, dim, block_size, halo_size, heads=4, mode='halo'):
        super(upsample, self).__init__()

        blocks = nn.ModuleList()
        if mode == 'halo':
            blocks.append(HaloAttention(
                dim=dim,  # dimension of feature map
                block_size=block_size,  # neighborhood block size (feature map must be divisible by this)
                halo_size=halo_size,  # halo size (block receptive field)
                dim_head=dim // heads,  # dimension of each head
                heads=heads,  # number of attention heads
                out_dim=dim
            ))
            blocks.append(nn.Conv2d(dim, 4 * dim, 1, 1, 0))
            blocks.append(nn.PixelShuffle(2))
        elif mode == 'conv':
            blocks.append(
                ConvBlock(in_channels=dim, out_channels=4 * dim, kernel_size=5, stride=1, padding=2, norm='in',
                          act='leakyrelu', padding_mode='replicate'))
            blocks.append(nn.PixelShuffle(2))
        elif mode == 'deconv':
            blocks.append(nn.ConvTranspose2d(dim, dim, kernel_size=6, padding=2, stride=2,
                                             padding_mode='zeros'))
        elif mode == 'interp':
            blocks.append(nn.Upsample(scale_factor=2.0, mode='bicubic'))

        self.blocks = blocks

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class downsample(nn.Module):
    def __init__(self, in_channels, out_channels, mode='pool'):
        super(downsample, self).__init__()
        blocks = nn.ModuleList()
        if mode == 'conv':
            blocks.append(nn.Conv2d(in_channels, out_channels, 3, 2, 1, padding_mode='replicate'))
            blocks.append(nn.InstanceNorm2d(out_channels))
            blocks.append(nn.LeakyReLU())
            blocks.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode='replicate'))
        elif mode == 'pool':
            blocks.append(nn.MaxPool2d(2))
        elif mode == 'interp':
            blocks.append(nn.Upsample(scale_factor=0.5, mode='bicubic'))
        self.blocks = blocks

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class scale_refine_block(nn.Module):
    def __init__(self, dim, block_size, halo_size, heads=4, upmode='conv', downmode='conv', noud=False):
        super(scale_refine_block, self).__init__()
        if noud is False:
            self.up = upsample(dim, block_size, halo_size, heads=heads, mode=upmode)
            self.down = downsample(dim, dim, mode=downmode)
        else:
            self.up = ConvBlock(in_channels = dim, out_channels = dim, kernel_size = 5, stride = 1, padding = 2, norm = 'in',
                 act = 'leakyrelu', padding_mode = 'reflect', spectral_norm = False, dilation = 1)
            self.down = ConvBlock(in_channels = dim, out_channels = dim, kernel_size = 5, stride = 1, padding = 2, norm = 'in',
                 act = 'leakyrelu', padding_mode = 'reflect', spectral_norm = False, dilation = 1)

    def forward(self, x):
        return self.down(self.up(x))


class patch_match(nn.Module):
    '''
    This module defines the learnable patch match, which consists of:
    1. feature extractor
        1.1 patch size
    2. attention map
        2.1 Sparse constraint
    3. soft attention
        3.1 aggregte
    '''

    def __init__(self, use_vgg=True):
        super(patch_match, self).__init__()
        # load vgg module
        Conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        Conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        relu = nn.ReLU()
        if use_vgg:
            _vgg = torch.load('Pretrained/vgg_conv.pth')
            Conv1.weight = nn.Parameter(_vgg['conv1_1.weight'])
            Conv1.bias = nn.Parameter(_vgg['conv1_1.bias'])
            Conv2.weight = nn.Parameter(_vgg['conv1_2.weight'])
            Conv2.bias = nn.Parameter(_vgg['conv1_2.bias'])
        self.feature_extractor = nn.Sequential(Conv1, relu, Conv2, relu)

    def forward(self, x, ref, fold_params={'kernel_size': (7, 7), 'padding': 3, 'stride': 1, 'dilation': 1},
                divisor=None, n=1, hard=False, skip = 1, tao = 0.01):
        x_feat = (x + 1.) / 2.
        ref = (ref + 1.) / 2.
        ref_unfold = F.unfold(ref, **fold_params)

        r_feat = self.feature_extractor(ref)[:,::skip,...]
        r_feat = r_feat.repeat([x_feat.shape[0], 1, 1, 1])
        ref_unfold = ref_unfold.repeat([x_feat.shape[0], 1, 1])

        refsr_lv1_unfold = F.unfold(r_feat, **fold_params)  # (N,CxP1xP2, HrxWr)
        refsr_lv1_unfold = F.normalize(refsr_lv1_unfold, dim=1)  # (N,CxP1xP2, HrxWr)
        for i in range(n):
            x_feat = self.feature_extractor(x_feat)[:,::skip,...]
            lrsr_lv1_unfold = F.unfold(x_feat, **fold_params)  # (N, CxP1xP2, HlxWl)
            lrsr_lv1_unfold = lrsr_lv1_unfold.permute(0, 2, 1)   # (N, HlxWl, CxP1xP2)
            lrsr_lv1_unfold = F.normalize(lrsr_lv1_unfold, dim=2, eps=1e-8)  # (N, HlxWl, CxP1xP2)
            R_lv1 = lrsr_lv1_unfold @ refsr_lv1_unfold  # [N, Hl*Wl, Hr*Wr]
            R_lv1 = F.softmax(R_lv1 / tao, dim=-1)  # [N, Hl*Wl, Hr*Wr] Get the attention map

            R_lv1 = einops.rearrange(R_lv1, 'b l r -> (b l) r')
            values, indices = torch.max(R_lv1, dim=-1)
            min_encodings = torch.zeros_like(R_lv1)
            min_encodings.scatter_(1, indices.unsqueeze(1), 1 if hard else values.unsqueeze(1))
            R_lv1 = einops.rearrange(min_encodings, '(b l) r -> b l r', b=x.shape[0])
            fusion = torch.bmm(ref_unfold,
                               R_lv1.permute(0, 2, 1))  # [N,3xP1xP2, HrxWr], [N, Hr*Wr, Hl*Wl] [N, 3xP1xP2, Hl*Wl]
            x_feat = F.fold(fusion, output_size=(x.size(2), x.size(3)),
                            **fold_params) * (divisor.expand(x.shape).to(x.device))
        x_feat = x_feat * 2 - 1
        return x_feat, R_lv1 if n > 0 else torch.zeros_like(x_feat)


class Discriminator(nn.Module):
    def __init__(self, size):
        super(Discriminator, self).__init__()
        dim = 32
        act = 'leakyrelu'
        norm = 'in'
        padding_norm = 'zeros'
        self.blocks = nn.ModuleList()
        self.blocks.append(
            ConvBlock(in_channels=3, out_channels=dim, stride=1, kernel_size=3,
                      padding=1, padding_mode=padding_norm, norm=norm, act=act, ))
        for i in range(3):
            block = ConvBlock(in_channels=2 ** i * dim, out_channels=2 ** (i + 1) * dim,
                              stride=2, kernel_size=3,
                              padding=1, padding_mode=padding_norm, norm=norm, act=act, )
            self.blocks.append(block)
            # block = ResBlock(in_channels = attn_dim, out_channels = attn_dim, kernel_size = 3,
            #                  stride = 1, padding = 1, padding_mode = padding_norm, norm = True)

            # self.blocks.append(nn.MaxPool2d(2))
        # self.blocks.append(
        #     nn.AdaptiveMaxPool2d((1, 1)))

        # estimate the output length
        self.register_buffer('prob', torch.randn(1, 3, *size))
        for block in self.blocks:
            self.prob = block(self.prob)

        self.DIM = functools.reduce(lambda x, y: x * y, self.prob.shape)
        # self.blocks.append(nn.Conv2d(8* dim, dim, 1, 1, 0))
        self.linear = nn.Linear(self.DIM, 1)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = x.view(-1, self.DIM)
        x = self.linear(x)
        return x


class Discriminator_Hr(nn.Module):
    def __init__(self):
        super(Discriminator_Hr, self).__init__()
        dim = 128
        act = 'leakyrelu'
        norm = 'in'
        padding_norm = 'zeros'
        spectral_norm = False
        self.blocks1 = nn.ModuleList()
        self.blocks1.append(
            ConvBlock(in_channels=3, out_channels=dim, stride=1, kernel_size=3,
                      padding=1, padding_mode=padding_norm, norm=norm, act=act,
                      spectral_norm = spectral_norm))
        for i in range(3):
            block = ConvBlock(in_channels=dim, out_channels=dim,
                              stride=1, kernel_size=3,
                              padding=1, padding_mode=padding_norm, norm=norm, act=act, spectral_norm = spectral_norm)
            self.blocks1.append(block)
        self.blocks1.append(nn.Conv2d(dim, 1, 3, 1, 1))


    def forward(self, x):
        for block in self.blocks1:
            x = block(x)

        return x

# class MS_Discriminator(nn.Module):
#     """ Discriminator of the GAN """
#
#     def __init__(self, stage = 2, dim = 128):
#         """
#         constructor for the class
#         :param depth: total depth of the discriminator
#                        (Must be equal to the Generator depth)
#         :param dim: size of the deepest features extracted
#                              (Must be equal to Generator latent_size)
#         :param use_eql: whether to use the equalized learning rate or not
#         :param gpu_parallelize: whether to use DataParallel on the discriminator
#                                 Note that the Last block contains StdDev layer
#                                 So, it is not parallelized.
#         """
#
#         super().__init__()
#
#         # create state of the object
#         self.stage = stage
#         original_dim = dim
#         spectral_norm = False
#         padding_mode = 'zeros'
#         self.head =  ConvBlock(in_channels = 3, out_channels = original_dim, kernel_size = 3, stride = 1, padding = 1,
#                           norm = 'in', act = 'leakyrelu', padding_mode = padding_mode,
#                           spectral_norm = spectral_norm,
#                           dilation = 1)
#         # create the fromRGB layers for various inputs:
#         for i in range(stage):
#             dim = int(dim / 2)
#             kernel_size = 1 + 2*(i+1)
#             dilation = 1
#             padding = kernel_size // 2
#             blocks = nn.Sequential(
#                 ConvBlock(in_channels = original_dim, out_channels = dim, kernel_size = 3, stride = 1,
#                           padding = 1,
#                           norm = 'in', act = 'leakyrelu', padding_mode = padding_mode, spectral_norm = spectral_norm,
#                           dilation = dilation),
#                 ConvBlock(in_channels = dim, out_channels = dim, kernel_size = kernel_size, stride = 1, padding = padding,
#                           norm = 'in', act = 'leakyrelu', padding_mode = padding_mode, spectral_norm = spectral_norm,
#                           dilation = dilation),
#                 ConvBlock(in_channels = dim, out_channels = dim, kernel_size = kernel_size, stride = 1, padding = padding,
#                           norm = 'in', act = 'leakyrelu', padding_mode = padding_mode, spectral_norm = spectral_norm,
#                           dilation = dilation),
#                 ConvBlock(in_channels = dim, out_channels = dim, kernel_size = kernel_size, stride = 1, padding = padding,
#                           norm = 'in', act = 'leakyrelu', padding_mode = padding_mode, spectral_norm = spectral_norm,
#                           dilation = dilation),
#                 nn.Conv2d(in_channels = dim, out_channels = 1, kernel_size = kernel_size, stride = 1, padding = padding, dilation = 1,
#                      padding_mode = padding_mode)
#             )
#             setattr(self, f'feature{i}', blocks)
#         setattr(self, f'feature{stage}', blocks)
#
#
#     def forward(self, x):
#         """
#         forward pass of the discriminator
#         :param inputs: (multi-scale input images) to the network list[Tensors]
#         :return: out => raw prediction values
#         """
#         y = 0
#         x = self.head(x)
#         for i in range(self.stage+1):
#             # if i > 0:
#             #     x += x_
#             y += getattr(self, f'feature{i}')(x)
#         return y

# class MS_Discriminator(nn.Module):
#     """ Discriminator of the GAN """
#
#     def __init__(self, stage = 2, dim = 128):
#         """
#         constructor for the class
#         :param depth: total depth of the discriminator
#                        (Must be equal to the Generator depth)
#         :param dim: size of the deepest features extracted
#                              (Must be equal to Generator latent_size)
#         :param use_eql: whether to use the equalized learning rate or not
#         :param gpu_parallelize: whether to use DataParallel on the discriminator
#                                 Note that the Last block contains StdDev layer
#                                 So, it is not parallelized.
#         """
#
#         super().__init__()
#
#         # create state of the object
#         self.stage = stage
#         original_dim = dim
#         spectral_norm = False
#         padding_mode = 'zeros'
#         self.head =  ConvBlock(in_channels = 3, out_channels = original_dim, kernel_size = 3, stride = 1, padding = 1,
#                           norm = 'in', act = 'leakyrelu', padding_mode = padding_mode,
#                           spectral_norm = spectral_norm,
#                           dilation = 1)
#         # create the fromRGB layers for various inputs:
#         for i in range(stage):
#             dim = original_dim
#             kernel_size = 3
#             dilation = 1
#             padding = kernel_size // 2
#             blocks = nn.Sequential(
#                 ConvBlock(in_channels = original_dim, out_channels = dim, kernel_size = 3, stride = 1,
#                           padding = 1,
#                           norm = 'in', act = 'leakyrelu', padding_mode = padding_mode, spectral_norm = spectral_norm,
#                           dilation = dilation),
#                 ConvBlock(in_channels = dim, out_channels = dim, kernel_size = kernel_size, stride = 1, padding = padding,
#                           norm = 'in', act = 'leakyrelu', padding_mode = padding_mode, spectral_norm = spectral_norm,
#                           dilation = dilation),
#                 ConvBlock(in_channels = dim, out_channels = dim, kernel_size = kernel_size, stride = 1, padding = padding,
#                           norm = 'in', act = 'leakyrelu', padding_mode = padding_mode, spectral_norm = spectral_norm,
#                           dilation = dilation),
#                 ConvBlock(in_channels = dim, out_channels = dim, kernel_size = kernel_size, stride = 1, padding = padding,
#                           norm = 'in', act = 'leakyrelu', padding_mode = padding_mode, spectral_norm = spectral_norm,
#                           dilation = dilation),
#                 nn.Conv2d(in_channels = dim, out_channels = 1, kernel_size = kernel_size, stride = 1, padding = padding, dilation = 1,
#                      padding_mode = padding_mode)
#             )
#             setattr(self, f'feature{i}', blocks)
#         setattr(self, f'feature{stage}', blocks)
#
#
#     def forward(self, x):
#         """
#         forward pass of the discriminator
#         :param inputs: (multi-scale input images) to the network list[Tensors]
#         :return: out => raw prediction values
#         """
#         y = 0
#         x = self.head(x)
#         for i in range(self.stage+1):
#             # if i > 0:
#             #     x += x_
#             y += getattr(self, f'feature{i}')(x)
#         return y

# class MS_Discriminator(nn.Module):
#     """ Discriminator of the GAN """
#
#     def __init__(self, stage = 2, dim = 128):
#         """
#         constructor for the class
#         :param depth: total depth of the discriminator
#                        (Must be equal to the Generator depth)
#         :param dim: size of the deepest features extracted
#                              (Must be equal to Generator latent_size)
#         :param use_eql: whether to use the equalized learning rate or not
#         :param gpu_parallelize: whether to use DataParallel on the discriminator
#                                 Note that the Last block contains StdDev layer
#                                 So, it is not parallelized.
#         """
#
#         super().__init__()
#
#         # create state of the object
#         self.stage = stage
#         self.dim = dim
#         spectral_norm = False
#         padding_mode = 'zeros'
#         self.factors = [0.5, 0.75, 1.0][-stage:]
#         for i in range(stage):
#             kernel_size = 3
#             dilation = 1
#             padding = 1
#             blocks = nn.Sequential(
#                 ConvBlock(in_channels = 3, out_channels = dim, kernel_size = 3, stride = 1, padding = 1,
#                           norm = 'in', act = 'leakyrelu', padding_mode = padding_mode, spectral_norm = spectral_norm,
#                           dilation = 1),
#                 ConvBlock(in_channels = dim, out_channels = dim, kernel_size = kernel_size, stride = 1, padding = padding,
#                           norm = 'in', act = 'leakyrelu', padding_mode = padding_mode, spectral_norm = spectral_norm,
#                           dilation = dilation),
#                 ConvBlock(in_channels = dim, out_channels = dim, kernel_size = kernel_size, stride = 1, padding = padding,
#                           norm = 'in', act = 'leakyrelu', padding_mode = padding_mode, spectral_norm = spectral_norm,
#                           dilation = dilation),
#                 ConvBlock(in_channels = dim, out_channels = dim, kernel_size = kernel_size, stride = 1, padding = padding,
#                           norm = 'in', act = 'leakyrelu', padding_mode = padding_mode, spectral_norm = spectral_norm,
#                           dilation = dilation),
#                 nn.Conv2d(in_channels = dim, out_channels = 1, kernel_size = kernel_size, stride = 1, padding = padding,
#                           padding_mode = padding_mode)
#             )
#             setattr(self, f'feature{i}', blocks)
#
#     def forward(self, input):
#         """
#         forward pass of the discriminator
#         :param inputs: (multi-scale input images) to the network list[Tensors]
#         :return: out => raw prediction values
#         """
#
#         for i in range(self.stage):
#             x = torch.nn.functional.interpolate(input, scale_factor = self.factors[i])
#             y += torch.mean(getattr(self, f'feature{i}')(x))
#         return y

class MS_Discriminator(nn.Module):
    """ Discriminator of the GAN """

    def __init__(self, stage = 2, dim = 128, spectral_norm = False):
        """
        constructor for the class
        :param depth: total depth of the discriminator
                       (Must be equal to the Generator depth)
        :param dim: size of the deepest features extracted
                             (Must be equal to Generator latent_size)
        :param use_eql: whether to use the equalized learning rate or not
        :param gpu_parallelize: whether to use DataParallel on the discriminator
                                Note that the Last block contains StdDev layer
                                So, it is not parallelized.
        """

        super().__init__()

        # create state of the object
        self.stage = stage
        self.dim = dim
        padding_mode = 'replicate'
        # create the fromRGB layers for various inputs:

        self.head = ConvBlock(in_channels = 3, out_channels = dim, kernel_size = 3, stride = 1, padding = 1,
                          norm = 'in', act = 'leakyrelu', padding_mode = padding_mode, spectral_norm = spectral_norm,
                          dilation = 1)

        for i in range(stage):
            kernel_size = 3
            dilation = 1 + i
            padding = dilation * (kernel_size // 2)
            blocks = nn.Sequential(
                ConvBlock(in_channels = dim, out_channels = dim, kernel_size = kernel_size, stride = 1, padding = padding,
                          norm = 'in', act = 'leakyrelu', padding_mode = padding_mode, spectral_norm = spectral_norm,
                          dilation = dilation),
                ConvBlock(in_channels = dim, out_channels = dim, kernel_size = kernel_size, stride = 1, padding = padding,
                          norm = 'in', act = 'leakyrelu', padding_mode = padding_mode, spectral_norm = spectral_norm,
                          dilation = dilation),
                ConvBlock(in_channels = dim, out_channels = dim, kernel_size = kernel_size, stride = 1, padding = padding,
                          norm = 'in', act = 'leakyrelu', padding_mode = padding_mode, spectral_norm = spectral_norm,
                          dilation = dilation),
                nn.Conv2d(in_channels = dim, out_channels = 1, kernel_size = 3, stride = 1, padding =1,
                          padding_mode = padding_mode),
            )
            setattr(self, f'k3feature{i}', blocks)

        # dim = dim // 2
        # for i in range(stage):
        #     kernel_size = 5
        #     dilation = 1 + i
        #     padding = dilation * (kernel_size // 2)
        #     blocks = nn.Sequential(
        #         ConvBlock(in_channels = dim*2, out_channels = dim, kernel_size = kernel_size, stride = 1, padding = padding,
        #                   norm = 'in', act = 'leakyrelu', padding_mode = padding_mode, spectral_norm = spectral_norm,
        #                   dilation = dilation),
        #         ConvBlock(in_channels = dim, out_channels = dim, kernel_size = kernel_size, stride = 1, padding = padding,
        #                   norm = 'in', act = 'leakyrelu', padding_mode = padding_mode, spectral_norm = spectral_norm,
        #                   dilation = dilation),
        #         ConvBlock(in_channels = dim, out_channels = dim, kernel_size = kernel_size, stride = 1, padding = padding,
        #                   norm = 'in', act = 'leakyrelu', padding_mode = padding_mode, spectral_norm = spectral_norm,
        #                   dilation = dilation),
        #         nn.Conv2d(in_channels = dim, out_channels = 1, kernel_size = 3, stride = 1, padding = 1,
        #                   padding_mode = padding_mode),
        #     )
        #     setattr(self, f'k5feature{i}', blocks)

    def forward(self, x):
        """
        forward pass of the discriminator
        :param inputs: (multi-scale input images) to the network list[Tensors]
        :return: out => raw prediction values
        """
        x = self.head(x)
        y = 0
        for i in range(self.stage):
            y += getattr(self, f'k3feature{i}')(x)
        # for i in range(self.stage):
        #     y += getattr(self, f'k5feature{i}')(x)
        return y

# class MS_Discriminator(nn.Module):
#     """ Discriminator of the GAN """
#
#     def __init__(self, stage = 2, dim = 128):
#         """
#         constructor for the class
#         :param depth: total depth of the discriminator
#                        (Must be equal to the Generator depth)
#         :param dim: size of the deepest features extracted
#                              (Must be equal to Generator latent_size)
#         :param use_eql: whether to use the equalized learning rate or not
#         :param gpu_parallelize: whether to use DataParallel on the discriminator
#                                 Note that the Last block contains StdDev layer
#                                 So, it is not parallelized.
#         """
#
#         super().__init__()
#
#         # create state of the object
#         self.stage = stage
#         self.dim = dim
#         spectral_norm = False
#         padding_mode = 'reflect'
#
#         self.blocks = nn.Sequential(
#             ConvBlock(in_channels = 3, out_channels = dim, kernel_size = 3, stride = 1, padding = 1,
#                       norm = 'in', act = 'leakyrelu', padding_mode = padding_mode, spectral_norm = spectral_norm,
#                       dilation = 1),
#             ConvBlock(in_channels = dim, out_channels = dim, kernel_size = 3, stride = 1, padding = 1,
#                       norm = 'in', act = 'leakyrelu', padding_mode = padding_mode, spectral_norm = spectral_norm,
#                       dilation = 1),
#             ConvBlock(in_channels = dim, out_channels = dim, kernel_size = 3, stride = 1, padding = 1,
#                       norm = 'in', act = 'leakyrelu', padding_mode = padding_mode, spectral_norm = spectral_norm,
#                       dilation = 1),
#             ConvBlock(in_channels = dim, out_channels = dim, kernel_size = 3, stride = 1, padding = 1,
#                       norm = 'in', act = 'leakyrelu', padding_mode = padding_mode, spectral_norm = spectral_norm,
#                       dilation = 1),
#
#         )
#
#         for i in range(stage):
#             kernel_size = 5
#             dilation = stage+1
#             padding = dilation*(2)
#             dis = nn.Conv2d(in_channels = dim, out_channels = 1, kernel_size = kernel_size, stride = 1, padding = padding, dilation = 1)
#             setattr(self, f'dis{i}', dis)
#
#
#
#     def forward(self, x):
#         """
#         forward pass of the discriminator
#         :param inputs: (multi-scale input images) to the network list[Tensors]
#         :return: out => raw prediction values
#         """
#         for block in self.blocks:
#             x = block(x)
#
#         y = 0
#         for i in range(self.stage):
#             y += getattr(self, f'dis{i}')(x)
#         return y

class Generator(nn.Module):
    '''
    input noise: z, (h, w)
    pass positional encoding
    pass conv blocks
    pass patch match
    pass to rgb
    return rgb image
    '''
    def __init__(self, in_channels=1, mid_channels=64, out_channels=3, kernel_size=3, layers=5, stride=1,
                 pos_enc='SPE',
                 norm='in', act='leakyrelu', padding_mode='zeros'):
        super(Generator, self).__init__()
        DIM = mid_channels
        self.use_pe = pos_enc
        if pos_enc == 'SPE':
            in_channels += 8
            self.pos_enc_func = functools.partial(PosEncoding.SinusoidalPositionalEncoding, d_model=8)
        elif pos_enc == 'CSG':
            in_channels += 2
            self.pos_enc_func = PosEncoding.CartesianSpatialGrid

        self.main = nn.Sequential(
            nn.Conv2d(in_channels, DIM, 3, 1),

            nn.Conv2d(DIM, 2*DIM, 3, 1),
            nn.InstanceNorm2d(2*DIM),
            nn.ReLU(True),

            nn.Conv2d(2*DIM, 2*DIM, 3, 1),
            nn.InstanceNorm2d(4*DIM),
            nn.ReLU(True),

            nn.Conv2d(2*DIM, 2*DIM, 3, 1),
            nn.InstanceNorm2d(2*DIM),
            nn.ReLU(True),

            nn.Conv2d(2*DIM, 3, 3, 1),
            nn.Tanh())

        self.pad = nn.ZeroPad2d(5)

    def forward(self, x):

        H, W = x.shape[-2:]
        pos = self.pos_enc_func(height=H, width=W).to(x.device)
        x = torch.cat([x, pos.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)], dim=1)
        x = self.pad(x)
        x = self.main(x)
        return x


class recnet(nn.Module):
    def __init__(self, dim, code_dim, updown_list, stages =3, downmode='conv', upmode='conv', noud = False):
        super(recnet, self).__init__()

        blocks = nn.ModuleList()
        blocks.append(nn.Conv2d(code_dim, dim, 1, 1, 0))
        blocks.append(nn.InstanceNorm2d(dim))
        blocks.append(nn.LeakyReLU())
        blocks.append(nn.Conv2d(dim, dim, 1, 1, 0))
        for i in range(stages):
            for _ in range(updown_list[i]):
                blocks.append(scale_refine_block(dim, 1, 0, heads=4, upmode=upmode, downmode=downmode, noud=noud))
            blocks.append(upsample(dim, 1, 0, mode=upmode))

        blocks.append(
            ConvBlocks(n=1, in_channels=dim, out_channels=dim // 2, padding_mode='replicate',
                       kernel_size=7,
                       padding=3, stride=1))
        blocks.append(nn.Conv2d(dim // 2, 3, 3, 1, 1))
        blocks.append(nn.Tanh())

        self.blocks = blocks

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# class hrnet(nn.Module):
#     def __init__(self, dim=64):
#         super(hrnet, self).__init__()
#         self.blocks = nn.ModuleList()
#         self.blocks.append(ConvBlock(in_channels = 3, out_channels = dim,
#                               stride = 1, kernel_size = 3,
#                               padding = 1, padding_mode = 'replicate', norm = 'in', act = 'leakyrelu',
#                               spectral_norm = False))
#         for i in range(3):
#             block = ConvBlock(in_channels = dim, out_channels = dim,
#                               stride = 1, kernel_size = 3,
#                               padding = 1, padding_mode = 'replicate', norm = 'in', act = 'leakyrelu',
#                               spectral_norm = False)
#             self.blocks.append(block)
#         self.blocks.append(nn.Conv2d(in_channels = dim, out_channels = 3,
#                                      stride = 1, kernel_size = 3,
#                                      padding = 1, padding_mode = 'replicate'))
#         self.blocks.append(nn.Tanh())
#     def forward(self, input):
#         x = input
#         for block in self.blocks:
#             x = block(x)
#         return x

class hrnet(nn.Module):
    def __init__(self, dim=64):
        super(hrnet, self).__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(ConvBlock(in_channels = 3, out_channels = dim,
                              stride = 1, kernel_size = 3,
                              padding = 1, padding_mode = 'replicate', norm = 'in', act = 'leakyrelu',
                              spectral_norm = False))
        for i in range(3):
            self.blocks.append(ConvBlock(in_channels = dim, out_channels = dim,
                                         stride = 1, kernel_size = 3,
                                         padding = 1, padding_mode = 'replicate', norm = 'in', act = 'leakyrelu',
                                         spectral_norm = False))

        self.blocks.append(upsample(dim, 2, 1, heads = 4, mode = 'conv'))
        for i in range(3):
            self.blocks.append(ConvBlock(in_channels = dim, out_channels = dim,
                                         stride = 1, kernel_size = 7,
                                         padding = 3, padding_mode = 'replicate', norm = 'in', act = 'leakyrelu',
                                         spectral_norm = False))
        self.blocks.append(nn.Conv2d(in_channels = dim, out_channels = 3,
                                     stride = 1, kernel_size = 7,
                                     padding = 3, padding_mode = 'replicate'))
        self.blocks.append(nn.Tanh())
    def forward(self, input):
        x = input
        for block in self.blocks:
            x = block(x)
        return x

class hrnet1024(nn.Module):
    def __init__(self, dim=64):
        super(hrnet1024, self).__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(ConvBlock(in_channels = 3, out_channels = dim,
                              stride = 1, kernel_size = 3,
                              padding = 1, padding_mode = 'replicate', norm = 'in', act = 'leakyrelu',
                              spectral_norm = False))
        for i in range(3):
            self.blocks.append(ConvBlock(in_channels = dim, out_channels = dim,
                                         stride = 1, kernel_size = 3,
                                         padding = 1, padding_mode = 'replicate', norm = 'in', act = 'leakyrelu',
                                         spectral_norm = False))

        self.blocks.append(upsample(dim, 2, 1, heads = 4, mode = 'conv'))
        for i in range(3):
            self.blocks.append(ConvBlock(in_channels = dim, out_channels = dim,
                                         stride = 1, kernel_size = 3,
                                         padding = 1, padding_mode = 'replicate', norm = 'in', act = 'leakyrelu',
                                         spectral_norm = False))
        self.blocks.append(upsample(dim, 2, 1, heads = 4, mode = 'conv'))
        self.blocks.append(nn.Conv2d(in_channels = dim, out_channels = 3,
                                     stride = 1, kernel_size = 7,
                                     padding = 3, padding_mode = 'replicate'))
        self.blocks.append(nn.Tanh())
    def forward(self, input):
        x = input
        for block in self.blocks:
            x = block(x)
        return x


class dbpn_docoder(nn.Module):
    def __init__(self, code_dim, dim, kernel_size=3, stride=2, padding=1, bias=True, activation='lrelu',
                 norm='instance', scale=2):
        import dbpn
        super(dbpn_docoder, self).__init__()
        self.head = nn.ModuleList()
        self.head.append(nn.Conv2d(code_dim, dim, 1, 1, 0))
        self.head.append(nn.InstanceNorm2d(dim))
        self.head.append(nn.LeakyReLU())
        self.head.append(nn.Conv2d(dim, dim, 1, 1, 0))

        self.body = nn.ModuleList()
        self.body.append(
            dbpn.UpBlockPix(dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, activation='lrelu',
                            norm='instance', scale=2))
        self.body.append(
            dbpn.UpBlockPix(dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, activation='lrelu',
                            norm='instance', scale=2))
        self.body.append(
            dbpn.UpBlockPix(dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, activation='lrelu',
                            norm='instance', scale=2))
        # x4
        self.tail = nn.ModuleList()
        self.tail.append(
            ConvBlocks(n=1, in_channels=dim, out_channels=dim // 2, padding_mode='replicate',
                       kernel_size=7,
                       padding=3, stride=1))
        self.tail.append(nn.Conv2d(dim // 2, 3, 1, 1, 0))
        self.tail.append(nn.Tanh())

    def forward(self, x):
        for block in self.head:
            x = block(x)
        for block in self.body:
            x = block(x)
        for block in self.tail:
            x = block(x)
        return x


# class ddbpn_docoder(nn.Module):
#     def __init__(self, dim, code_dim, updown_list, downsample_mode = 'conv', upsample_mode = 'pixelshuffle'):
#         super(ddbpn_docoder, self).__init__()
#         blocks = nn.ModuleList()
#         blocks.append(nn.Conv2d(code_dim, dim, 1, 1, 0))
#         blocks.append(nn.InstanceNorm2d(dim))
#         blocks.append(nn.LeakyReLU())
#         blocks.append(nn.Conv2d(dim, dim, 1, 1, 0))
#         self.head = blocks
#         # # x1
#         # blocks.append(downsample(dim, dim, mode = downsample_mode))
#         # blocks.append(upsample(dim, 1, 0, mode = 'pixelshuffle'))
#         # blocks.append(downsample(dim, dim, mode = downsample_mode))
#         # blocks.append(upsample(dim, 1, 0, mode = 'pixelshuffle'))
#         # x2
#         self.up1 = upsample(dim, 1, 0, mode = upsample_mode)
#         self.down1 = downsample(dim, dim, mode = downsample_mode)
#
#         self.up2 = upsample(dim, 1, 0, mode = upsample_mode)
#         self.down2 = downsample(dim, dim, mode = downsample_mode)
#
#         self.up3 = upsample(dim, 1, 0, mode = upsample_mode)
#         self.down3 = D_DownBlock(base_filter, kernel, stride, padding, 3)
#         self.up4 = D_UpBlock(base_filter, kernel, stride, padding, 3)
#         self.down4 = D_DownBlock(base_filter, kernel, stride, padding, 4)
#         self.up5 = D_UpBlock(base_filter, kernel, stride, padding, 4)
#         self.down5 = D_DownBlock(base_filter, kernel, stride, padding, 5)
#         self.up6 = D_UpBlock(base_filter, kernel, stride, padding, 5)
#         self.down6 = D_DownBlock(base_filter, kernel, stride, padding, 6)
#         self.up7 = D_UpBlock(base_filter, kernel, stride, padding, 6)
#
#         # to rgb
#
#         blocks.append(
#             ConvBlocks(n=1, in_channels=dim, out_channels=dim // 2, padding_mode='replicate',
#                        kernel_size=7,
#                        padding=3, stride=1))
#         blocks.append(nn.Conv2d(dim // 2, 3, 1, 1, 0))
#         blocks.append(nn.Tanh())
#
#         self.blocks = blocks
#
#     def forward(self, x):
#         for block in self.blocks:
#             x = block(x)
#
#         h1 = self.up1(x)
#         l1 = self.down1(h1)
#         h2 = self.up2(l1)
#
#         concat_h = torch.cat((h2, h1), 1)
#         l = self.down2(concat_h)
#
#         concat_l = torch.cat((l, l1), 1)
#         h = self.up3(concat_l)
#
#         concat_h = torch.cat((h, concat_h), 1)
#         l = self.down3(concat_h)
#
#         concat_l = torch.cat((l, concat_l), 1)
#         h = self.up4(concat_l)
#
#         concat_h = torch.cat((h, concat_h), 1)
#         l = self.down4(concat_h)
#
#         concat_l = torch.cat((l, concat_l), 1)
#         h = self.up5(concat_l)
#
#         concat_h = torch.cat((h, concat_h), 1)
#         l = self.down5(concat_h)
#
#         concat_l = torch.cat((l, concat_l), 1)
#         h = self.up6(concat_l)
#
#         concat_h = torch.cat((h, concat_h), 1)
#         l = self.down6(concat_h)
#
#         concat_l = torch.cat((l, concat_l), 1)
#         h = self.up7(concat_l)
#
#         concat_h = torch.cat((h, concat_h), 1)
#         x = self.output_conv(concat_h)


if __name__ == '__main__':
    pm = patch_match().float()
    x = functions.read_image(r'Input/Images/colusseum.png')
    x = functions.resize(x, size=[128, 128])
    ref = x
    fold_params = {'kernel_size': (3, 3), 'padding': 1, 'stride': 1, 'dilation': 1}
    divisor = functions.getDivisor(x, fold_params)
    x_pm, attn = pm(x, ref, fold_params=fold_params, divisor=divisor, n=1)
    functions.show_image(x_pm)
