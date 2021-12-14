import os

import torch
import torch.nn as nn

import Utils.functions as functions
from Models.models import scale_refine_block, upsample
from Models.blocks import ConvBlocks, ConvBlock
import functools

class PyramidGenerator():

    def __init__(self, Gs, reals, NoiseAmp, opt):

        self.Gs = Gs
        self.reals = reals
        self.NoiseAmp = NoiseAmp
        self.opt = opt

        kernel_size = 7
        self.fold_params = {'kernel_size': (kernel_size, kernel_size), 'padding': kernel_size // 2, 'stride': 1,
                       'dilation': 1}
        self.divisor = functions.getDivisor(self.reals[-2], self.fold_params)
        from Models.TTSR import TTSR
        self.ttsr = TTSR().to(opt.device)
        self.ref =  torch.cat([self.reals[len(Gs)- 2],torch.flip(self.reals[len(Gs)- 2], dims = [-1])], dim = -1)

    def __call__(self, StartHeight = 0, EndHeight = None, PrevImage = None, FixedNoise = None, SavePath = None,
                 batch_size = 1, start_noise = None):

        '''PrevImage: has not upsampled'''

        if StartHeight == 0:
            PrevImage = torch.zeros_like(self.reals[0])
        else:
            assert PrevImage is not None

        EndHeight = len(self.Gs) - 1 if EndHeight is None else EndHeight
        assert (EndHeight + 1) <= len(self.Gs)

        if SavePath is not None:
            if not os.path.exists(SavePath):
                os.makedirs(SavePath)

        for height in range(StartHeight, EndHeight + 1):
            G = self.Gs[height]
            real = self.reals[height]
            if isinstance(G, PEStructGenerator):
                G.eval()
                if FixedNoise is None:
                    if start_noise is not None:
                        noise = start_noise
                    else:
                        noise = torch.randn(batch_size, 1, *real.shape[-2:]).to(self.opt.device)
                    PrevImage = G(noise)
                else:
                    noise = FixedNoise[height].to(self.opt.device)
                    PrevImage = G(noise)
                    continue

            elif isinstance(G, DipGenerator):
                # PrevImage = functions.upsampling(PrevImage, real.shape[2], real.shape[3])
                # PrevImage, _ = self.ttsr(PrevImage, self.ref , self.ref,
                #      fold_params = self.fold_params,
                #      divisor = self.divisor, n = 1, lv = 1, skip = 4, return_img = True)
                PrevImage = G(PrevImage.detach())
            else:
                noiseAmp = self.NoiseAmp[height]
                if FixedNoise is None:
                    noise = functions.generate_noise(real.shape[1:], num_samp = batch_size, device = self.opt.device)
                else:
                    noise = FixedNoise[height].to(self.opt.device)
                PrevImage = functions.upsampling(PrevImage, real.shape[2], real.shape[3])
                PrevImage = G(PrevImage, noise, noiseAmp)
        return PrevImage



class PEStructGenerator(nn.Module):

    def __init__(self, opt):
        super(PEStructGenerator, self).__init__()
        DIM = opt.struct_channel_dim
        self.use_pe = opt.use_pe

        if self.use_pe == 'CSG':
            from Utils.PosEncoding import CartesianSpatialGrid
            self.pe = CartesianSpatialGrid
            self.input_dim = 3
        elif self.use_pe == 'SPE':
            from Utils.PosEncoding import SinusoidalPositionalEncoding
            self.pe = functools.partial(SinusoidalPositionalEncoding, d_model=8)
            self.input_dim = 1 + opt.pe_dim
        elif self.use_pe is None:
            self.input_dim = 1

        self.main = nn.Sequential(
            nn.Conv2d(self.input_dim, DIM, 3, 1),

            nn.Conv2d(DIM, 2 * DIM, 3, 1),
            nn.InstanceNorm2d(2 * DIM),
            nn.ReLU(True),

            nn.Conv2d(2 * DIM, 2 * DIM, 3, 1),
            nn.InstanceNorm2d(4 * DIM),
            nn.ReLU(True),

            nn.Conv2d(2 * DIM, 2 * DIM, 3, 1),
            nn.InstanceNorm2d(2 * DIM),
            nn.ReLU(True),

            nn.Conv2d(2 * DIM, 3, 3, 1),
            nn.Tanh())

        self.pad = nn.ZeroPad2d(5)

    def forward(self, x):
        H, W = x.shape[-2:]
        pos = self.pe(height = H, width = W).to(x.device)
        x = torch.cat([x, pos.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)], dim = 1)
        x = self.pad(x)
        x = self.main(x)
        return x

class StructDiscriminator(nn.Module):
    def __init__(self, size):
        super(StructDiscriminator, self).__init__()
        dim = 32
        act = 'leakyrelu'
        norm = 'in'
        padding_norm = 'zeros'
        self.blocks = nn.ModuleList()
        self.blocks.append(
            ConvBlock(in_channels = 3, out_channels = dim, stride = 1, kernel_size = 3,
                      padding = 1, padding_mode = padding_norm, norm = norm, act = act, ))
        for i in range(3):
            block = ConvBlock(in_channels = 2 ** i * dim, out_channels = 2 ** (i + 1) * dim,
                              stride = 2, kernel_size = 3,
                              padding = 1, padding_mode = padding_norm, norm = norm, act = act, )
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


class SemanticGenerator(nn.Module):
    def __init__(self, opt):
        super(SemanticGenerator, self).__init__()
        self.head = ConvBlock(in_channels = opt.nc_im, out_channels = 64, kernel_size = 3, stride = 1, padding = 0,
                              norm = 'in',
                              act = 'leakyrelu', padding_mode = 'zeros', spectral_norm = False,
                              dilation = 1) 
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            block = ConvBlock(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 0,
                              norm = 'in',
                              act = 'leakyrelu', padding_mode = 'zeros', spectral_norm = False, dilation = 1)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Sequential(
            nn.Conv2d(64, opt.nc_im, kernel_size = 3, stride = 1, padding = 0),
            nn.Tanh()
        )

        self.pad = nn.ZeroPad2d(opt.num_layer)

    def forward(self, img, noise, alpha):
        x = self.pad(img + noise * alpha)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


class SemanticDiscriminator(nn.Module):
    def __init__(self, opt):
        super(SemanticDiscriminator, self).__init__()
        self.head = ConvBlock(in_channels = opt.nc_im, out_channels = 64, kernel_size = 3, padding = 1, stride = 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            block = ConvBlock(64, 64, 3, 1, 1)
            self.body.add_module('block%d' % (i + 1), block)

        self.tail = nn.Conv2d(64, 1, 3, 1, 1)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x




class DipGenerator(nn.Module):
    def __init__(self, dim, code_dim, updown_list, stages = 2, downmode = 'conv', upmode = 'conv', noud = False):
        super(DipGenerator, self).__init__()

        blocks = nn.ModuleList()
        blocks.append(nn.Conv2d(code_dim, dim, 1, 1, 0))
        blocks.append(nn.InstanceNorm2d(dim))
        blocks.append(nn.LeakyReLU(inplace = True, negative_slope = 0.2))
        blocks.append(nn.Conv2d(dim, dim, 1, 1, 0))
        for i in range(stages):
            for _ in range(updown_list[i]):
                blocks.append(
                    scale_refine_block(dim, 1, 0, heads = 4, upmode = upmode, downmode = downmode, noud = noud))
            blocks.append(upsample(dim, 1, 0, mode = upmode))

        blocks.append(
            ConvBlocks(n = 1, in_channels = dim, out_channels = dim // 2, padding_mode = 'reflect',
                       kernel_size = 7,
                       padding = 3, stride = 1))
        blocks.append(nn.Conv2d(dim // 2, 3, 3, 1, 1))
        blocks.append(nn.Tanh())

        self.blocks = blocks

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x




class DipDiscriminator(nn.Module):
    def __init__(self, stage = 2, dim = 64):
        super(DipDiscriminator, self).__init__()
        self.stage = stage
        self.dim = dim
        spectral_norm = False
        padding_mode = 'reflect'
        # create the fromRGB layers for various inputs:

        self.head = ConvBlock(in_channels = 3, out_channels = dim, kernel_size = 3, stride = 1, padding = 1,
                              norm = 'in', act = 'leakyrelu', padding_mode = padding_mode,
                              spectral_norm = spectral_norm,
                              dilation = 1)

        for i in range(stage):
            kernel_size = 3
            dilation = 1 + stage
            padding = 2
            blocks = nn.Sequential(
                ConvBlock(in_channels = dim, out_channels = dim, kernel_size = kernel_size, stride = 1,
                          padding = padding,
                          norm = 'in', act = 'leakyrelu', padding_mode = padding_mode, spectral_norm = spectral_norm,
                          dilation = dilation),
                ConvBlock(in_channels = dim, out_channels = dim, kernel_size = kernel_size, stride = 1,
                          padding = padding,
                          norm = 'in', act = 'leakyrelu', padding_mode = padding_mode, spectral_norm = spectral_norm,
                          dilation = dilation),
                ConvBlock(in_channels = dim, out_channels = dim, kernel_size = kernel_size, stride = 1,
                          padding = padding,
                          norm = 'in', act = 'leakyrelu', padding_mode = padding_mode, spectral_norm = spectral_norm,
                          dilation = dilation),
                nn.Conv2d(in_channels = dim, out_channels = 1, kernel_size = 3, stride = 1, padding = 1,
                          padding_mode = padding_mode),
            )
            setattr(self, f'k3feature{i}', blocks)

        dim = dim // 2
        for i in range(stage):
            kernel_size = 5
            dilation = 1 + stage
            padding = 2
            blocks = nn.Sequential(
                ConvBlock(in_channels = dim * 2, out_channels = dim, kernel_size = kernel_size, stride = 1,
                          padding = padding,
                          norm = 'in', act = 'leakyrelu', padding_mode = padding_mode, spectral_norm = spectral_norm,
                          dilation = dilation),
                ConvBlock(in_channels = dim, out_channels = dim, kernel_size = kernel_size, stride = 1,
                          padding = padding,
                          norm = 'in', act = 'leakyrelu', padding_mode = padding_mode, spectral_norm = spectral_norm,
                          dilation = dilation),
                ConvBlock(in_channels = dim, out_channels = dim, kernel_size = kernel_size, stride = 1,
                          padding = padding,
                          norm = 'in', act = 'leakyrelu', padding_mode = padding_mode, spectral_norm = spectral_norm,
                          dilation = dilation),
                nn.Conv2d(in_channels = dim, out_channels = 1, kernel_size = 3, stride = 1, padding = 1,
                          padding_mode = padding_mode),
            )
            setattr(self, f'k5feature{i}', blocks)

    def forward(self, x):
        """
        forward pass of the discriminator
        :param inputs: (multi-scale input images) to the network list[Tensors]
        :return: out => raw prediction values
        """
        x = self.head(x)
        y = 0
        # for i in range(self.stage):
        #     y += getattr(self, f'k3feature{i}')(x)
        for i in range(self.stage):
            y += getattr(self, f'k5feature{i}')(x)
        return y


class TextureGenerator(nn.Module):
    def __init__(self, opt):
        super(TextureGenerator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        self.head = ConvBlock(in_channels = opt.nc_im, out_channels = 64, kernel_size = 3, stride = 1, padding = 0, norm = 'in',
                 act = 'leakyrelu', padding_mode = 'zeros', spectral_norm = False, dilation = 1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            block = ConvBlock(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 0, norm = 'in',
                 act = 'leakyrelu', padding_mode = 'zeros', spectral_norm = False, dilation = 1)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Sequential(
            nn.Conv2d(64, opt.nc_im, kernel_size=3, stride=1, padding=0),
            nn.Tanh()
        )

        self.pad = nn.ZeroPad2d(opt.num_layer)

    def forward(self, input, noise, noiseAmp):
        x = self.pad(input+ noiseAmp*noise)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x + input

class TextureDiscriminator(nn.Module):
    def __init__(self, opt):
        super(TextureDiscriminator, self).__init__()
        self.head = ConvBlock(in_channels = opt.nc_im, out_channels = 64, kernel_size = 3, padding = 1, stride = 1)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            block = ConvBlock(64, 64, 3, 1, 1)
            self.body.add_module('block%d' % (i + 1), block)

        self.tail = nn.Conv2d(64, 1, 3, 1, 1)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x
