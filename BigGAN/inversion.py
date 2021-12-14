# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 16:59:05 2020

@author: Administrator
"""
import sys
from BigGAN.models.biggan import Generator, Discriminator
import Utils.utils as utils
import Utils.loss as loss
import torch
from torch.autograd import Variable
import time
import torchvision
import torch.nn.functional as F
import os
import math
from Pretrained import vgg


class Inversion(object):
    z_star: Variable
    iterations: tuple

    def __init__(self, config):
        r'''
        config:
            gen_mode
            weight_mode
        '''
        self.z_grad = []
        self.record_loss = []
        self.config = config
        self.gan_mode = config['gan_mode']
        self.random_G = config['random_G']
        self.weights_root = config['weights_root']
        self.load_weights = config['load_weights']
        self.use_ema = config['use_ema']
        self.ftr_type = config['ftr_type']
        self.ftr_num = config['ftr_num']
        self.image_path = config['image_path']
        self.iterations = config['iterations']
        self.y = torch.tensor([config['class']])
        self.resolution = config['resolution']
        self.dim_z = config['dim_z']
        self.G_lr = config['G_lr']  # init biggan
        self.G_lrs = config['G_lrs']
        self.z_lrs = config['z_lrs']
        self.lambda_LK = config['lambda_LK']
        self.lambda_MSE = config['lambda_MSE']
        self.lambda_D = config['lambda_D']
        self.lambda_P = config['lambda_P']

        self.vgg_use_layers = ['r33', 'r43', 'r53']

        self.task_mode = 'gen'
        self.img = None
        self.gpu_ids = [0]
        self.ftpg = config['ftpg']
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.record = {'MSE': [], 'LK': [], 'DLoss': [], 'Loss': [], 'grad': [], 'P': [], 'config': config}
        self.warm_up = 0

        self.z = torch.Tensor(self.dim_z).unsqueeze(0)

    def set_generator(self):
        self.G = Generator(**self.config).cuda()  # **返回dict， *返回tuple

        if self.random_G:
            self.G.init_weights()
        else:
            utils.load_weights(
                self.G if not self.use_ema else None,
                None,
                self.weights_root,
                name_suffix = self.load_weights,
                G_ema = self.G if self.use_ema else None,
                strict = False)
        utils.to_device(self.G, self.gpu_ids)
        self.G.eval()

    def set_discriminator(self):
        self.D = Discriminator(
            **self.config).cuda() if self.ftr_type == 'Discriminator' else None

        utils.load_weights(
            None,
            self.D,
            self.weights_root,
            name_suffix = self.load_weights,
            G_ema = None,
            strict = False)
        utils.to_device(self.D, self.gpu_ids)
        # bn               self.D_lr.eval()

    def set_image(self):
        self.image_name = os.path.basename(self.image_path).split('.')[0]
        self.path_ = os.path.join('JitData', self.image_name)
        self.save_path = os.path.join(self.path_, time.strftime('%Y-%m-%d#%H#%M#%S', time.localtime(time.time())))

        # You can read img from image path, or use existing image
        if self.img is None:
            original_image = utils.pil_loader(self.image_path)
        else:
            original_image = self.img
        self.original_image = utils.preprocessor(original_image, self.resolution).to(self.device)

        if self.lambda_P != 0:
            self.vgg = vgg.VGG().to(self.device)
            self.vgg.load_state_dict(torch.load('vgg/' + 'vgg_conv.pth'))

    def set_target(self):
        if self.task_mode == 'gen':
            self.degraded_image = None
        else:
            self.degraded_image = None
            pass

    def random_z(self):
        return self.z.normal_(mean = 0, std = 0.3).to(self.device)

    def select_z(self, num_z = 100):
        min_loss = 1e8
        for i in range(num_z):
            self.z = self.random_z()
            if self.y < 0:
                self.y = self.y.random_(0, self.config['n_classes'])
            self.y = self.y.to(self.device)

            self.forward(self.z, self.y)
            temp_loss, temp_prior = self.d_loss()
            temp_loss = temp_loss.item()
            if temp_loss <= min_loss:
                min_loss = temp_loss
                self.z_star = self.z
                print(i)

        self.z_star_temp = self.z_star.clone()
        self.z_star = Variable(self.z_star, requires_grad = True)

    def set_optimizer(self):
        self.optimizer_z = torch.optim.Adam([{'params': self.z_star}], lr = self.z_lrs[0], betas = [0.9, 0.999])
        self.scheduler_z = utils.LRScheduler(self.optimizer_z, self.warm_up)
        self.optimizer_G = torch.optim.Adam([{'params': self.G.parameters()}], lr = self.G_lrs[0], betas = [0.9, 0.999])
        self.scheduler_G = utils.LRScheduler(self.optimizer_G, self.warm_up)

    def d_loss(self):
        return loss.DiscriminatoryLoss(self.D, self.original_image, self.temp_x, self.y, self.ftr_num[0])

    def mse_loss(self):
        return F.mse_loss(self.original_image, self.temp_x)

    def perceptual_loss(self):
        self.vgg_feature1 = self.vgg(self.original_image.detach(), self.vgg_use_layers)
        self.vgg_feature2 = self.vgg(self.temp_x, self.vgg_use_layers)
        return sum(
            [F.l1_loss(self.vgg_feature1[i], self.vgg_feature2[i]) for i in range(len(self.vgg_use_layers))]) / len(
            self.vgg_use_layers)

    def likelihood_loss(self):
        return (self.z_star ** 2 / 2).mean()

    def inversion_loss(self):
        self.MSE = self.mse_loss()
        self.LK = self.likelihood_loss()
        self.DLoss, self.score = self.d_loss()
        self.Loss = 0
        if self.lambda_P != 0:
            self.PLoss = self.perceptual_loss()
            self.Loss += self.lambda_P * self.PLoss
        if self.lambda_D != 0:
            self.Loss += self.lambda_D * self.DLoss
        if self.lambda_MSE != 0:
            self.Loss += self.lambda_MSE * self.MSE
        if self.lambda_LK != 0:
            self.Loss += self.lambda_LK * self.LK

    def forward(self, z, y):
        self.temp_x = self.G(z, self.G.shared(y))

    def Record(self):
        self.record['Loss'].append(self.Loss.item())
        self.record['MSE'].append(self.MSE.item())
        self.record['LK'].append(self.LK.item())
        self.record['DLoss'].append(self.DLoss.item())
        # self.record['P'].append((self.PLoss.item()))
        self.record['grad'].append(torch.sqrt((self.z_star.grad ** 2).sum()).item())

    def inversion(self):

        start_time = time.time()
        count = 0

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        for stage, iterations in enumerate(self.iterations):
            for iteration in range(iterations):
                self.scheduler_G.update(count, self.G_lrs[stage],
                                        self.ftpg[stage])
                self.scheduler_z.update(count, self.z_lrs[stage])

                self.optimizer_z.zero_grad()
                self.optimizer_G.zero_grad()

                self.forward(self.z_star, self.y)
                self.inversion_loss()
                self.Loss.backward()

                self.optimizer_z.step()
                self.optimizer_G.step()

                self.Record()

                count += 1
                if iteration % 10 == 0:
                    print(
                        f'[stage {stage}][{iteration}/{iterations}]: loss {self.Loss.item():.3f}, time {time.time() - start_time:.3f}')
                    print(f'likelihood {self.LK.item():.3f}, MSE {self.MSE.item():.3f}, DLoss {self.DLoss.item():.3f}')
                    # torchvision.Utils.save_image(self.temp_x, f'{self.save_path}/{stage}_{iteration}.png',normalize=True)
                    start_time = time.time()
                    # img_grid = torchvision.Utils.make_grid(self.G(self.z_star, self.G.shared(self.y)))

        print(f'stage {stage}[{iterations}/{iterations}]: loss {self.Loss.item()}, time {time.time() - start_time:.3f}')
        print(f'likelihood {self.LK.item():.3f}, MSE {self.MSE.item():.3f}, DLoss {self.DLoss.item():.3f}')
        # torchvision.Utils.save_image(self.temp_x, f'{self.save_path}/{stage}_{iteration}.png', normalize=True)
        torch.save(self.record, f'{self.save_path}/record.pth')
        print(">>>>>>END<<<<<<<")

    def random_sample(self, n = 64):
        z_star = self.z_star.clone()
        z_rand = torch.ones_like(z_star)
        flag = 0
        with torch.no_grad():
            for i in range(n):
                # add random noise to the latent vector
                z_rand.normal_()
                self.forward(z_rand, self.y)
                score, _ = self.D(self.temp_x, self.y)
                if flag == 0:
                    Rand = self.temp_x.cpu()
                else:
                    Rand = torch.cat((Rand, self.temp_x.cpu()), dim = 0)
                flag += 1
                torchvision.utils.save_image(self.temp_x.cpu(), f'{self.save_path}/random_{i}_score{score.item()}.png',
                                             normalize = True)
            torchvision.utils.save_image(Rand, f'{self.save_path}/randoms.png', nrow = int(math.sqrt(n)),
                                         normalize = True)

    def jitter(self, stds = [0.1, 0.2, 0.3, 0.4, 0.5], jitter_num = 100, save_path = None):
        if save_path is not None:
            self.save_path = save_path
        z_star = self.z_star.clone()
        z_rand = torch.ones_like(z_star)
        flag = 0
        n = 8
        with torch.no_grad():
            for std in stds:
                for i in range(jitter_num):
                    # add random noise to the latent vector
                    z_rand.normal_()
                    z = z_star + std * z_rand
                    self.forward(z, self.y)
                    # score, _ = self.D_lr(self.temp_x, self.y)
                    if flag == 0:
                        Jitter = self.temp_x.cpu()
                    else:
                        Jitter = torch.cat((Jitter, self.temp_x.cpu()), dim = 0)
                    flag += 1
                    torchvision.utils.save_image(self.temp_x.cpu(), f'{self.save_path}/jitter_{std}_{i}.png',
                                                 normalize = True)
