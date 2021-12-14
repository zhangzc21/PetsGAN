import math
import os
import torch
import torch.nn as nn
import torchvision as tv
from PIL import Image
import random
import functools
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

resize = functools.partial(F.interpolate, mode = 'bicubic', align_corners = True)


def show_image(x):
    x = ((x + 1) / 2).clamp(0, 1)
    x = tv.utils.make_grid(x, normalize = False)
    x = x.detach().cpu().permute(1, 2, 0).numpy()
    plt.imshow(x)
    plt.show()


def read_image(path):
    x = Image.open(path)
    x = x.convert("RGB")
    composed = tv.transforms.Compose(
        [tv.transforms.ToTensor(), tv.transforms.Normalize(mean = [0.5] * 3, std = [0.5] * 3)])
    x = composed(x).unsqueeze(0)
    return x


def create_reals_pyramid(real, opt):
    opt.scale_factor = math.pow(opt.min_size / (min(real.shape[2], real.shape[3])),
                                1 / (opt.pyramid_height))
    print(f'use pyramid height to calculate scale factor, scale factor = {opt.scale_factor}')
    reals = []
    shapes = []
    min_shape = (
        real.shape[2] * opt.scale_factor ** opt.pyramid_height, real.shape[3] * opt.scale_factor ** opt.pyramid_height)
    if opt.rescale_method == 'petsgan':
        opt.scale_factor = math.pow(opt.min_size / (min(real.shape[2] // 4, real.shape[3] // 4)),
                                    1 / (opt.pyramid_height - 1))
    # min_shape = (
    #     real.shape[2] * opt.scale_factor ** opt.pyramid_height, real.shape[3] * opt.scale_factor ** opt.pyramid_height)
    for i in range(opt.pyramid_height):
        if opt.rescale_method == 'consingan':
            scale = math.pow(opt.scale_factor,
                             ((opt.pyramid_height - 1) / math.log(opt.pyramid_height)) * math.log(
                                 opt.pyramid_height - i) + 1)
            curr_real = resize(real, scale_factor = scale)
        elif opt.rescale_method == 'singan':
            scale = math.pow(opt.scale_factor, opt.pyramid_height - i)
            curr_real = resize(real, scale_factor = scale)
            shape = curr_real.shape[-2:]
        elif opt.rescale_method == 'exsingan':
            r = 1 / opt.scale_factor - 1
            up_scale = (1 + i * r + (i * (i - 1) * (i - 2) / 2) * (r ** 3))
            shape = (int(min_shape[0] * up_scale), int(min_shape[1] * up_scale))
            curr_real = resize(real, shape)
        elif opt.rescale_method == 'petsgan':
            r = 1 / opt.scale_factor - 1
            up_scale = (1 + i * r + (i * (i - 1) * (i - 2) / 2) * (r ** 3))
            shape = (int(min_shape[0] * up_scale), int(min_shape[1] * up_scale))
            curr_real = resize(real, shape)
            if i == opt.pyramid_height - 1:
                shape = [real.shape[2] // 4, real.shape[3] // 4]
                curr_real = resize(real, shape)
        reals.append(curr_real)
        shapes.append(curr_real.shape[-2:])
        print(f'pyramid {i}:', shape[-2:])
    reals.append(real)
    shapes.append(real.shape[-2:])
    opt.shapes = shapes
    print(f'pyramid {i + 1}:', real.shape[-2:])
    return reals


def create_pyramid(real, opt):
    reals = []
    shapes = []
    # min_shape = (
    #     real.shape[2] * opt.scale_factor ** opt.pyramid_height, real.shape[3] * opt.scale_factor ** opt.pyramid_height)
    min_shape = (
        real.shape[2] // 8, real.shape[3] // 8)
    for i in range(opt.pyramid_height):
        if opt.rescale_method == 'consingan':
            scale = math.pow(opt.scale_factor,
                             ((opt.pyramid_height - 1) / math.log(opt.pyramid_height)) * math.log(
                                 opt.pyramid_height - i) + 1)
            curr_real = resize(real, scale_factor = scale)
        elif opt.rescale_method == 'singan':
            scale = math.pow(opt.scale_factor, opt.pyramid_height - i)
            curr_real = resize(real, scale_factor = scale)
        elif opt.rescale_method == 'exsingan':
            r = 1 / opt.scale_factor - 1
            up_scale = (1 + i * r + (i * (i - 1) * (i - 2) / 2) * (r ** 3))
            shape = (int(min_shape[0] * up_scale), int(min_shape[1] * up_scale))
            curr_real = resize(real, shape)
        reals.append(curr_real.to(opt.device))
        shapes.append(curr_real.shape[-2:])
    reals.append(real)
    shapes.append(real.shape[-2:])
    return reals, shapes


def generate_noise(size, num_samp = 1, device = 'cuda', type = 'gaussian', scale = 1):
    if type == 'gaussian':
        noise = torch.randn(num_samp, size[0], round(size[1] / scale), round(size[2] / scale), device = device)
        noise = upsampling(noise, size[1], size[2])
    elif type == 'gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device = device) + 5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device = device)
        noise = noise1 + noise2
    elif type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], device = device)
    else:
        raise NotImplementedError
    return noise


def upsampling(im: torch.Tensor, sx, sy):
    m = nn.Upsample(size = [round(sx), round(sy)], mode = 'bilinear', align_corners = True)
    return m(im)


def calc_gradient_penalty_fc(netD, real_data, fake_data):
    # right
    alpha = torch.randn(real_data.shape[0], 1, 1, 1).to(real_data.device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = torch.autograd.Variable(interpolates, requires_grad = True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs = disc_interpolates, inputs = interpolates,
                                    grad_outputs = torch.ones(disc_interpolates.size()).to(real_data.device),
                                    create_graph = True, retain_graph = True, only_inputs = True)[0]
    gradients = gradients.contiguous().view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim = 1) - 1) ** 2).mean() * 10

    return gradient_penalty


def calc_gradient_penalty_conv(netD, real_data, fake_data):
    # right
    alpha = torch.randn(real_data.shape[0], 1, 1, 1).to(real_data.device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = torch.autograd.Variable(interpolates, requires_grad = True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs = disc_interpolates, inputs = interpolates,
                                    grad_outputs = torch.ones(disc_interpolates.size()).to(real_data.device),
                                    create_graph = True, retain_graph = True, only_inputs = True)[0]
    # gradients = gradients.contiguous().view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim = 1) - 1) ** 2).mean() * 0.1

    return gradient_penalty


def R1Penalty(real_img, f, patch = False):
    # gradient penalty
    real_img = torch.autograd.Variable(real_img, requires_grad = True)
    prediction_real = f(real_img)
    grad_real = torch.autograd.grad(outputs = prediction_real.sum(), inputs = real_img, create_graph = True)[0]
    # Calc regularization
    regularization_loss: torch.Tensor = 0.2 \
                                        * grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
    return regularization_loss


def getDivisor(x, fp):
    img = torch.ones_like(x)
    unfold_img = F.unfold(img, **fp)
    fold_img = F.fold(unfold_img, output_size = x.shape[-2:], **fp)
    divisor = 1 / fold_img
    divisor[fold_img == 0] = 0
    return divisor


def save_image(img, root, nrow = 6, ):
    img = ((img + 1) / 2).clamp(0, 1)
    tv.utils.save_image(img, root, normalize = False, nrow = nrow)


def generate_image(frame, netG, device = 'cpu'):
    fixed_noise_128 = torch.randn(36, 128).to(device)
    samples = netG(fixed_noise_128)
    samples = samples.view(-1, 3, 32, 32)
    samples = samples.mul(0.5).add(0.5)
    samples = samples.cpu().data
    path = './Temp/samples_{}.png'.format(frame)
    if not os.path.exists('Temp'):
        os.makedirs('Temp')
        tv.utils.save_image(samples, path, nrow = 6)


def getDivisor(x, fp):
    img = torch.ones_like(x)
    unfold_img = F.unfold(img, **fp)
    fold_img = F.fold(unfold_img, output_size = x.shape[-2:], **fp)
    divisor = 1 / fold_img
    divisor[fold_img == 0] = 0
    return divisor


class ema(object):
    def __init__(self, source, target, decay = 0.9999, start_itr = 0):
        self.source = source
        self.target = target
        self.decay = decay
        # Optional parameter indicating what iteration to start the decay at
        self.start_itr = start_itr
        # Initialize target's params to be source's
        self.source_dict = self.source.state_dict()
        self.target_dict = self.target.state_dict()
        print('Initializing G_EMA parameters to be source parameters...')
        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(self.source_dict[key].data)
                # target_dict[key].data = source_dict[key].data # Doesn't work!

    def update(self, itr = None):
        # If an iteration counter is provided and itr is less than the start itr,
        # peg the ema weights to the underlying weights.
        if itr and itr < self.start_itr:
            decay = 0.0
        else:
            decay = self.decay
        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(self.target_dict[key].data * decay
                                                 + self.source_dict[key].data * (1 - decay))


def seed_all(seed, dedeterministic = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if dedeterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def adjust_scales2image(real_, opt):
    if opt.pyramid_height is None:
        opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]),
                         1)  # min(256/max([real_.shape[0],real_.shape[1]]),1)
        real = resize(real_, scale_factor = opt.scale1)
        opt.scale_factor_init = opt.scale_factor
        opt.pyramid_height = math.ceil(
            (math.log(math.pow(opt.min_size / (min(real.shape[2], real.shape[3])), 1), opt.scale_factor_init))) + 1
    else:
        opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]),
                         1)  # min(256/max([real_.shape[0],real_.shape[1]]),1)
        real = resize(real_, scale_factor = opt.scale1)
        real = resize(real, size = [real.shape[2] // 8 * 8, real.shape[3] // 8 * 8])
        opt.pyramid_layers = opt.pyramid_height + 1
    return real
