import kornia
import torch
import torch.nn as nn
import pathlib
from glob import glob
import argparse
import torchsummary
import functools
import torch.nn.functional as F
from tqdm import tqdm
import random
from Utils import PrepareData, utils, functions, losses
import shutil, os
import time
import math
import logging

logging.basicConfig(level = logging.INFO)


class LRScheduler(object):

    def __init__(self, optimizer, warm_up):
        super(LRScheduler, self).__init__()
        self.optimizer = optimizer
        self.warm_up = warm_up

    def update(self, iteration, learning_rate, num_group = 1000, ratio = 1):
        if iteration < self.warm_up:
            learning_rate *= iteration / self.warm_up
        for i, param_group in enumerate(self.optimizer.param_groups):
            if i >= num_group:
                param_group['lr'] = 0
            else:
                param_group['lr'] = learning_rate * ratio ** i


class PetsGAN():
    def __init__(self, opt):

        self.opt = opt
        for key in opt.__dict__.keys():
            setattr(self, key, getattr(opt, key))

        self.save_dir = pathlib.Path(opt.save_dir)

        self.stages = int(math.log2(self.downscale_factor))
        self.use_flip = self.opt.use_flip
        self.resize = functools.partial(F.interpolate, mode = 'bicubic', align_corners = True)
        self.gan_start_epoch = 0
        self.recnet_start_epoch = 0
        self.co_start_epoch = 0
        self.fixed_noise = None

        image_name_wo_suffix = pathlib.Path(opt.image_name).stem
        self.image_path = pathlib.Path(opt.input_dir) / opt.image_name
        self.save_dir = pathlib.Path(opt.save_dir) / image_name_wo_suffix / opt.model_name
        self._make_dirs()

    def _copy_weights(self, model_name, gan = False, lpm = False, recnet = False, co = False):

        def _copy_weight(ref_path, save_path, key):
            if ref_path.exists():
                if not save_path.exists():
                    # os.remove(str(save_path))
                    shutil.copy(str(ref_path), str(save_path))
            else:
                logging.info(f'Ref {key} does not exist. \n')

        if gan:
            ref_path = self.save_dir.parent / model_name / 'weights' / 'gan.pth'
            save_path = self.save_dir / 'weights' / 'gan.pth'
            _copy_weight(ref_path, save_path, 'gan')
        if lpm:
            ref_path = self.save_dir.parent / model_name / 'weights' / 'lpm.pth'
            save_path = self.save_dir / 'weights' / 'lpm.pth'
            _copy_weight(ref_path, save_path, 'lpm')
        if recnet:
            ref_path = self.save_dir.parent / model_name / 'weights' / 'recnet.pth'
            save_path = self.save_dir / 'weights' / 'recnet.pth'
            _copy_weight(ref_path, save_path, 'recnet')
        if co:
            ref_path = self.save_dir.parent / model_name / 'weights' / 'co_recnet.pth'
            save_path = self.save_dir / 'weights' / 'co_recnet.pth'
            _copy_weight(ref_path, save_path, 'co_recnet')

    def _init_all(self, gan = True, pm = True, recnet = True, co = True, load_gan = True, load_lpm = True,
                  load_recnet = True, load_co = True):

        self._read_image()

        self.fold_params = {'kernel_size': (7, 7), 'padding': 3, 'stride': 1, 'dilation': 1}
        self.divisor = functions.getDivisor(self.lr_image, self.fold_params)

        self._save_code()
        if gan is True:
            self._define_generator()
            self._define_discriminator()
        if pm is True:
            self._define_patch_match()
        if recnet is True:
            self._define_recnet()
        self._define_optimizer(gan = gan, pm = pm, recnet = recnet)

        self.co_recnet = None
        if co is True:
            ''' define networks'''
            self._define_co_G()
            self._define_co_recnet()
            self._define_discriminator_hr()
            ''' define optimizer'''
            num_blocks = len(list(self.co_recnet.blocks))
            index = [0]
            ############################################################
            for i in range(num_blocks):
                if self.co_recnet.blocks[i]._get_name() == 'upsample':
                    index.append(i + 1)
            index.append(num_blocks)
            assert len(index) == self.stages + 2
            lr_rates = iter(self.lr_rates)
            lr_dict = []
            lr_dict.append(
                {'params': self.co_G.parameters(), 'lr': next(lr_rates)})
            for start, end in zip(index[:-1], index[1:]):
                lr = next(lr_rates)
                for i in range(start, end):
                    lr_dict.append({'params': self.co_recnet.blocks[i].parameters(), 'lr': lr})

            self.optimizer_co_recnet = torch.optim.Adam(lr_dict)
            self.optimizer_D_hr = torch.optim.Adam(self.D_hr.parameters(), lr = self.lr_D_hr)
            ''' load pre-trained '''
            self.sch_co_recnet = LRScheduler(self.optimizer_co_recnet, warm_up = 0)

        self._try_load_weight(load_gan = load_gan, load_lpm = load_lpm, load_recnet = load_recnet, load_co = load_co)

        self._define_loss()

    def _read_image(self):
        '''
        :return:
        self.lr_image
        self.hr_image
        self.lr_h
        self.lr_w
        self.hr_h
        self.hr_w
        '''

        real = functions.read_image(str(self.image_path)).to(self.device)
        self.origin = real
        functions.save_image(real, str(self.save_dir) + '/origin.jpg', nrow = 1)
        H, W = real.shape[2], real.shape[3]
        scale_factor = min([256 / H, 256 / W, 1])
        real = self.resize(real, scale_factor = scale_factor)
        real = self.resize(real, size = (real.shape[-2] // self.downscale_factor * self.downscale_factor,
                                         real.shape[-1] // self.downscale_factor * self.downscale_factor))
        self.hr_image = real.to(self.device)
        self.hr_h, self.hr_w = real.shape[-2], real.shape[-1]
        self.lr_h, self.lr_w = real.shape[-2] // self.downscale_factor, real.shape[-1] // self.downscale_factor
        self.lr_image = self.resize(self.hr_image, size = [self.lr_h, self.lr_w])
        self.ref_image = torch.cat([self.lr_image, torch.flip(self.lr_image, dims = [-1])], dim = -1)

    def _make_dirs(self):
        '''
        make path for saving
        '''
        self.save_code_dir = self.save_dir / 'codes'
        self.save_recnet_dir = self.save_dir / 'stage_rec'
        self.save_gan_dir = self.save_dir / 'stage_gan'
        self.save_weight_dir = self.save_dir / 'weights'
        self.save_final_syntheses_dir = self.save_dir / 'final_syntheses'
        self.save_co_dir = self.save_dir / 'stage_co'
        self.save_others = self.save_dir / 'others'

        self.save_code_dir.mkdir(parents = True, exist_ok = True)
        self.save_recnet_dir.mkdir(parents = True, exist_ok = True)
        self.save_gan_dir.mkdir(parents = True, exist_ok = True)
        self.save_weight_dir.mkdir(parents = True, exist_ok = True)
        self.save_final_syntheses_dir.mkdir(parents = True, exist_ok = True)
        self.save_others.mkdir(parents = True, exist_ok = True)
        self.save_co_dir.mkdir(parents = True, exist_ok = True)

    def _save_code(self):
        code_files = sorted(glob("./**/*.py", recursive = True))
        code_files = [path for path in code_files if 'Result' not in path]
        for cf in code_files:
            dst = f"{str(self.save_code_dir)}/{os.path.dirname(cf)}"
            os.makedirs(dst, exist_ok = True)
            shutil.copy(cf, dst)
        opt_dict = vars(self.opt)
        torch.save(opt_dict, f'{str(self.save_code_dir)}/opt_dict.pt')

    def _define_generator(self):
        from Models.models import Generator
        self.G = Generator(in_channels = 1, mid_channels = 64, out_channels = 3, kernel_size = 3, layers = 5,
                           stride = 1, pos_enc = 'SPE', norm = 'in', act = 'leakyrelu', padding_mode = 'reflect')
        self.G.to(self.device)

        if self.use_ema is True:
            import copy
            self.G_ema = copy.deepcopy(self.G)
            self.G_ema.to(self.device)
            self.G_EMA = functions.ema(source = self.G, target = self.G_ema, decay = 0.99, start_itr = 0)

        logging.debug("Generator:")
        logging.debug(torchsummary.summary(self.G, input_size = (1, self.lr_h, self.lr_w)))

    def _define_discriminator(self):
        from Models.models import Discriminator
        self.D = Discriminator(size = [self.lr_w, self.lr_h])
        self.D.to(self.device)

        logging.debug('Discriminator: \n')
        logging.debug(torchsummary.summary(self.D, input_size = (3, self.lr_h, self.lr_w,)))

    def _define_recnet(self):
        from Models.models import recnet
        self.recnet = recnet(dim = self.recnet_dim, code_dim = self.code_dim, downmode = self.downsample_mode,
                             stages = self.stages,
                             upmode = self.upsample_mode, updown_list = self.updown_list, noud = self.no_updown)
        self.recnet.to(self.device)

        logging.debug('recnet: \n')
        logging.debug(torchsummary.summary(self.recnet,
                                           input_size = (self.code_dim, self.lr_h, self.lr_w)))

    def _define_co_G(self):
        from Models.models import Generator
        self.co_G = Generator(in_channels = 1, mid_channels = 64, out_channels = 3, kernel_size = 3, layers = 5,
                              stride = 1, pos_enc = 'SPE', norm = 'in', act = 'leakyrelu', padding_mode = 'reflect')
        self.co_G.to(self.device)

        if self.use_ema is True:
            import copy
            self.co_G_ema = copy.deepcopy(self.co_G)
            self.co_G_ema.to(self.device)
            self.co_G_EMA = functions.ema(source = self.co_G, target = self.co_G_ema, decay = 0.99, start_itr = 0)

        with utils.HiddenPrints() if self.hiddenprints else utils.Prints():
            print("Co generator: \n")
            torchsummary.summary(self.co_G, input_size = (1, self.lr_h, self.lr_w))

    def _define_co_recnet(self):
        from Models.models import recnet
        self.co_recnet = recnet(dim = self.recnet_dim, code_dim = self.code_dim, downmode = self.downsample_mode,
                                stages = self.stages,
                                upmode = self.upsample_mode, updown_list = self.updown_list, noud = self.no_updown)
        self.co_recnet.to(self.device)
        if self.use_ema is True:
            import copy
            self.co_recnet_ema = copy.deepcopy(self.co_recnet)
            self.co_recnet_ema.to(self.device)
            self.co_recnet_EMA = functions.ema(source = self.co_recnet, target = self.co_recnet_ema, decay = 0.99,
                                               start_itr = 0)
        with utils.HiddenPrints() if self.hiddenprints else utils.Prints():
            print('Co recnet: \n')
            torchsummary.summary(self.recnet,
                                 input_size = (self.code_dim, self.lr_h, self.lr_w))

    def _define_discriminator_hr(self):
        from Models.models import MS_Discriminator
        self.D_hr = MS_Discriminator(stage = self.msd_stage, dim = self.msd_dim)
        self.D_hr.to(self.device)

        with utils.HiddenPrints() if self.hiddenprints else utils.Prints():
            print('Discriminator HR: \n')
            torchsummary.summary(self.D_hr, input_size = (3, self.hr_h, self.hr_w))

    def _define_patch_match(self):
        from Models.models import patch_match
        self.lpm = patch_match().float()
        self.lpm.to(self.device)

    def _define_optimizer(self, gan = True, pm = True, recnet = True):
        if gan is True:
            self.optimizer_G = torch.optim.Adam(
                [{'params': self.G.parameters(), 'lr': self.lr_g}])
            self.optimizer_D = torch.optim.Adam(
                [{'params': self.D.parameters(), 'lr': self.lr_d}])
        if pm is True:
            self.optimizer_lpm = torch.optim.Adam(
                [{'params': self.lpm.parameters(), 'lr': self.lr_lpm}])
        if recnet is True:
            self.optimizer_recnet = torch.optim.Adam(
                [{'params': self.recnet.parameters(), 'lr': self.lr_recnet}])
            self.lr_scheduler_recnet = torch.optim.lr_scheduler.StepLR(self.optimizer_recnet, 500, gamma = 0.5,
                                                                       last_epoch = -1, )

    def _define_loss(self):
        self.discriminator_loss = losses.loss_wgan_dis
        self.generator_loss = losses.loss_wgan_gen
        self.sparsity_loss = losses.loss_spa
        self.rec_loss = nn.MSELoss()
        self.hr_discriminator_loss = losses.loss_hinge_dis
        self.hr_generator_loss = losses.loss_hinge_gen

    def _try_load_weight(self, load_gan = True, load_lpm = True, load_recnet = True, load_co = True):
        if load_gan:
            try:
                pretrained_gan = torch.load(f'{str(self.save_weight_dir)}/gan.pth')
                self.G.load_state_dict(pretrained_gan['G'])
                self.D.load_state_dict(pretrained_gan['D_lr'])
                self.gan_start_epoch = pretrained_gan['epoch']
                self.optimizer_G.load_state_dict(pretrained_gan['optimizer_G'])
                self.optimizer_D.load_state_dict(pretrained_gan['optimizer_D'])
                self.fixed_noise = pretrained_gan['fixed_noise']
                print('pretrained gan exists.')
                if self.use_ema is True:
                    self.G_ema.load_state_dict(pretrained_gan['G_ema'])

            except OSError:
                print('pretrained gan dose not exist.')

        if load_lpm:
            try:
                pretrained_lpm = torch.load(f'{str(self.save_weight_dir)}/lpm.pth')
                self.lpm.load_state_dict(pretrained_lpm['lpm'])
                self.optimizer_lpm.load_state_dict(pretrained_lpm['optimizer_lpm'])
                print('pretrained lpm exists.')
            except OSError:
                print('pretrained lpm dose not exist.')

        if load_recnet:
            try:
                pretrained_recnet = torch.load(f'{str(self.save_weight_dir)}/recnet.pth')
                self.recnet.load_state_dict(pretrained_recnet['recnet'])
                self.recnet_start_epoch = pretrained_recnet['epoch']
                self.optimizer_recnet.load_state_dict(pretrained_recnet['optimizer_recnet'])
                self.lr_scheduler_recnet.load_state_dict(pretrained_recnet['lr_scheduler_recnet'])
                print('pretrained recnet exists.')
            except OSError:
                print('pretrained recnet dose not exist.')

        self.co_recnet_flag = 'Not Load'
        if load_co:
            try:
                pretrained_co_recnet = torch.load(f'{str(self.save_weight_dir)}/co_recnet.pth')
                self.co_G.load_state_dict(pretrained_co_recnet['co_G'])
                self.co_G_ema.load_state_dict(pretrained_co_recnet['co_G_ema'])
                self.co_recnet.load_state_dict(pretrained_co_recnet['co_recnet'])
                self.co_recnet_ema.load_state_dict(pretrained_co_recnet['co_recnet_ema'])
                self.co_start_epoch = pretrained_co_recnet['epoch']
                if self.co_start_epoch < self.co_epoch:
                    self.D_hr.load_state_dict(pretrained_co_recnet['D_hr'])
                    self.optimizer_D_hr.load_state_dict(pretrained_co_recnet['optimizer_D_hr'])
                    self.optimizer_co_recnet.load_state_dict(pretrained_co_recnet['optimizer_co_recnet'])
                self.co_recnet_flag = 'Load'
                print('pretrained co_recnet exists.')
            except OSError:
                print('pretrained co_recnet dose not exist.')

    def _prepare_data(self):
        # GAN inversion
        from BigGAN.RunJitter import RunJitter
        self.JitDataRoot = 'JitData/' + self.image_name.split('.')[0]
        if not os.path.exists(self.JitDataRoot):
            os.makedirs(self.JitDataRoot)
        if len(os.listdir(self.JitDataRoot)) == 0:
            RunJitter('%s/%s' % (self.input_dir, self.image_name), stds = [0.1, 0.2, 0.3, 0.4, 0.5], num_sample = 100,
                      cls = -1)
        from Utils.dataset import JitDataset

        Loader = JitDataset(self.JitDataRoot, image_size = (self.lr_h, self.lr_w))
        DataNum = Loader.__len__()
        DataSet = torch.utils.data.DataLoader(Loader, batch_size = DataNum, shuffle = True)
        DataAll = next(iter(DataSet)).to(self.device)
        DataAll = torch.cat([DataAll, torch.flip(DataAll, dims = [-1])], dim = 0)
        DataNum = DataAll.shape[0]
        DT = PrepareData.DataTrans()
        augment_num = int(self.mix_ratio / (1 - self.mix_ratio) * DataNum)
        for i in range(augment_num):
            DataAll = torch.cat(
                [DataAll, self.resize(DT.gen(self.hr_image), size = [self.lr_h, self.lr_w]).to(self.device)], dim = 0)
        # original image
        DataAll = torch.cat([DataAll, self.lr_image.expand(25,
                                                           *self.lr_image.shape[-3:])], dim = 0)
        DataAll = torch.cat([DataAll, torch.flip(self.lr_image, dims = [-1]).expand(25,
                                                                                    *self.lr_image.shape[-3:])],
                            dim = 0)

        DataNum = DataAll.shape[0]
        All_Data = DataAll[torch.randperm(DataNum, dtype = torch.int64)].to(self.device)
        return All_Data

    def internal_learning(self):
        start_epoch = self.recnet_start_epoch
        end_epoch = self.recnet_epoch
        save_epoch = self.recnet_save_epoch
        # noise = self.test_noise
        if start_epoch == end_epoch:
            return

        DT = PrepareData.DataTrans()
        lr_g = DT.gen(self.hr_image).to(self.device)
        for i in range(0):
            lr_g = torch.cat([lr_g, DT.gen(self.hr_image).to(self.device)], dim = 0)
        lr_g = self.resize(lr_g, size = [self.lr_h, self.lr_w]).to(self.device)
        lr_g_noise = (lr_g) + 0.3 * torch.randn_like(lr_g) + 0.3 * (torch.rand_like(lr_g) - 0.5)
        fold_params = {'kernel_size': (7, 7), 'padding': 3, 'stride': 1, 'dilation': 1}
        divisor = functions.getDivisor(self.lr_image, fold_params)

        if self.use_flip:
            hr_data = [self.hr_image, torch.flip(self.hr_image, dims = [-1])]
            lr_data = [self.lr_image, torch.flip(self.lr_image, dims = [-1])]
        else:
            hr_data = [self.hr_image]
            lr_data = [self.lr_image]
        EPOCHS = tqdm(range(start_epoch, end_epoch))
        color_jitter = kornia.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1)
        blur = kornia.augmentation.RandomGaussianBlur((7, 7), (1.0, 1.0), p = 0.5)
        noiseaug = kornia.augmentation.RandomGaussianNoise(0, 0.3, p = 0.5)
        for epoch in EPOCHS:
            self.recnet.train()
            self.lpm.train()
            self.optimizer_recnet.zero_grad()
            self.optimizer_lpm.zero_grad()

            index = random.randint(0, len(hr_data) - 1)
            hr_real, lr_real = hr_data[index], lr_data[index]

            lr_noise = (lr_real) + 0.01 * torch.randn_like(lr_real)
            lr_pm, attn = self.lpm(lr_noise, self.ref_image, fold_params = fold_params, divisor = divisor, n = 1)
            lr_rec_loss = self.rec_loss(lr_pm, lr_real)
            sparse_loss = self.sparsity_loss(attn) + 1
            rec = self.recnet(noiseaug(lr_real))
            hr_rec_loss = self.rec_loss(rec, hr_real)
            total_loss = hr_rec_loss + lr_rec_loss + sparse_loss

            total_loss.backward()
            self.optimizer_lpm.step()
            self.optimizer_recnet.step()
            self.lr_scheduler_recnet.step()

            EPOCHS.set_postfix(total_loss = total_loss.item(), hr_rec_loss = hr_rec_loss.item(),
                               lr_rec_loss = lr_rec_loss.item(), sparse_loss = sparse_loss.item())
            if (epoch + 1) % save_epoch == 0 or epoch == 0:
                self.recnet.eval()
                self.lpm.eval()
                functions.save_image(rec,
                                     '{}/rec_hr.jpg'.format(str(self.save_recnet_dir), epoch + 1))
                functions.save_image(lr_pm,
                                     '{}/rec_lrpm.jpg'.format(str(self.save_recnet_dir), epoch + 1))
                self._save_result(epoch, save_recnet = True, save_lpm = True)
                lr_g_pm, _ = self.lpm(lr_g_noise, self.ref_image, fold_params = fold_params, divisor = divisor,
                                      n = self.lpm_iter,
                                      hard = False)
                lr_g_pm_hard, _ = self.lpm(lr_g_noise, self.ref_image, fold_params = fold_params, divisor = divisor,
                                           n = self.lpm_iter,
                                           hard = True)
                save_lr = torch.cat([lr_g, lr_g_pm, lr_g_pm_hard], dim = 0)
                hr = self.recnet(lr_g)
                hr = torch.cat(
                    [hr, self.recnet(lr_g_pm)], dim = 0)
                hr = torch.cat(
                    [hr, self.recnet(lr_g_pm_hard)], dim = 0)
                functions.save_image(hr,
                                     '{}/hr_{}.jpg'.format(str(self.save_recnet_dir), epoch + 1),
                                     nrow = lr_g.shape[0])
                functions.save_image(save_lr,
                                     '{}/lr_{}.jpg'.format(str(self.save_recnet_dir), epoch + 1), nrow = lr_g.shape[0])

    def external_learning(self):
        start_epoch = self.gan_start_epoch
        end_epoch = self.gan_epoch
        save_epoch = self.gan_save_epoch
        if start_epoch == end_epoch:
            return

        if self.fixed_noise is None:
            self.fixed_noise = torch.randn(1, 1, self.lr_h, self.lr_w).to(self.device)
        noise = torch.randn(7, 1, self.lr_h, self.lr_w).to(self.device)
        _noise = torch.cat([noise, self.fixed_noise], dim = 0)

        fold_params = {'kernel_size': (7, 7), 'padding': 3, 'stride': 1, 'dilation': 1}
        divisor = functions.getDivisor(self.lr_image, fold_params)
        batch = self.gan_batch
        EPOCHS = tqdm(range(start_epoch, end_epoch))
        All_Data = self._prepare_data()
        # All_Data = self.lr_image
        #############
        for epoch in EPOCHS:
            ############################
            # (1) Update D_lr network
            ###########################

            errG = torch.Tensor([0]).to(self.device)
            gradient_penalty = torch.Tensor([0]).to(self.device)
            rec_loss = torch.Tensor([0]).to(self.device)
            diversity_loss = torch.Tensor([0]).to(self.device)
            self.D.train()
            self.G.train()
            self.G_ema.train()

            for i in range(self.D_iter):
                self.optimizer_D.zero_grad()
                index = torch.randint(0, All_Data.shape[0], (batch,), dtype = torch.int64)
                real_data = All_Data[index].detach()

                output_real = self.D(real_data)

                noise = torch.randn(batch, 1, self.lr_h, self.lr_w).to(self.device)
                with torch.no_grad():
                    fake = self.G(noise)
                output_fake = self.D(fake)
                errD_real, errD_fake = self.discriminator_loss(output_fake, output_real)
                gradient_penalty = functions.calc_gradient_penalty_fc(self.D, real_data, fake)
                # gradient_penalty = functions.R1Penalty(real_data, self.D_lr)
                errD_total = errD_real + errD_fake + gradient_penalty
                errD_total.backward()
                self.optimizer_D.step()

            for i in range(self.G_iter):
                self.optimizer_G.zero_grad()
                noise = torch.randn(batch, 1, self.lr_h, self.lr_w).to(self.device)
                new_code = self.G(noise)
                fake = new_code
                output = self.D(fake)
                errG = self.generator_loss(output)
                # diverse loss
                diversity_loss = -F.l1_loss(output[0], output[1])

                reconstruction = self.G(self.fixed_noise.to(self.device))
                rec_loss = 50 * F.mse_loss(reconstruction, self.lr_image)

                (errG + rec_loss + diversity_loss).backward()
                self.optimizer_G.step()
                if self.use_ema:
                    self.G_EMA.update()
            EPOCHS.set_postfix(errG = errG.item(), errD_fake = errD_fake.item(), errD_real = errD_real.item(),
                               gradient_penalty = gradient_penalty.item(), rec_loss = rec_loss.item(),
                               div_loss = diversity_loss.item())
            if (epoch + 1) % save_epoch == 0 or epoch == 0:
                self.G.eval()
                self.G_ema.eval()
                self.D.eval()
                self._save_result(epoch, save_gan = True)
                fake = self.G(_noise)
                fake_ = [self.G_ema(_noise)] if self.use_ema else []
                fake_ = torch.cat([fake] + fake_, dim = 0)
                save_lr = fake_
                functions.save_image(save_lr, f'{str(self.save_gan_dir)}/lr_{epoch + 1}.png',
                                     nrow = fake.shape[0])

    def co_learning(self):
        '''
        use internal learning network as internal prior
        train low resolution GAN coupled with internal learning prior
        '''
        tao = 0.00001
        if self.co_recnet_flag == 'Not Load':
            pretrained_recnet = torch.load(f'{str(self.save_weight_dir)}/recnet.pth')
            self.co_recnet.load_state_dict(pretrained_recnet['recnet'])
            self.co_recnet_ema.load_state_dict(pretrained_recnet['recnet'])

            pretrained_recnet = torch.load(f'{str(self.save_weight_dir)}/gan.pth')
            self.co_G.load_state_dict(pretrained_recnet['G_ema'])
            self.co_G_ema.load_state_dict(pretrained_recnet['G_ema'])

        start_epoch = self.co_start_epoch
        end_epoch = self.co_epoch
        save_epoch = self.gan_save_epoch
        batch = self.co_batch
        if start_epoch == end_epoch:
            return

        noise = torch.randn(5, 1, self.lr_h, self.lr_w).to(self.device)
        _noise = torch.cat([noise, self.fixed_noise], dim = 0)
        EPOCHS = tqdm(range(start_epoch, end_epoch))
        # weight_loss = weight_norm(self.recnet, self.co_recnet)

        w_loss = torch.tensor([0.0]).to(self.device)
        scale_factors = [0.8, 0.9, 1.0]

        if self.use_flip:
            real_data_lr = [self.lr_image, torch.flip(self.lr_image, dims = [-1])]
            real_data = [self.hr_image, torch.flip(self.hr_image, dims = [-1])]
        else:
            real_data_lr = [self.lr_image]
            real_data = [self.hr_image]

        blur = kornia.augmentation.RandomGaussianBlur((7, 7), (1, 1), p = 0.5)
        noiseaug = kornia.augmentation.RandomGaussianNoise(0, 0.3, p = 0.5)
        #############
        for epoch in EPOCHS:
            ############################
            # (1) Update D_lr network
            ###########################

            errG = torch.Tensor([0]).to(self.device)
            gradient_penalty = torch.Tensor([0]).to(self.device)
            fd_loss = torch.Tensor([0]).to(self.device)
            self.D_hr.train()
            self.co_G.train()
            self.co_recnet.train()

            for i in range(self.co_D_iter):
                index = random.randint(0, len(real_data) - 1)
                self.optimizer_D_hr.zero_grad()
                output_real_hr = self.D_hr(real_data[index])
                noise = torch.randn(batch, 1, self.lr_h, self.lr_w).to(self.device)
                with torch.no_grad():
                    fake_lr, _ = self.lpm(self.co_G(noise), self.ref_image, fold_params = self.fold_params,
                                          divisor = self.divisor,
                                          n = self.lpm_iter,
                                          hard = self.lpm_hard, tao = tao)
                    fake_hr = self.co_recnet(fake_lr)
                output_fake_hr = self.D_hr(fake_hr)
                errD_real_hr, errD_fake_hr = self.hr_discriminator_loss(output_fake_hr, output_real_hr)
                # gradient_penalty = functions.R1Penalty(self.hr_image, self.D_hr, patch = True)
                errD_total_hr = errD_real_hr + errD_fake_hr + gradient_penalty
                errD_total_hr.backward()
                self.optimizer_D_hr.step()

            for i in range(self.co_G_iter):
                index = random.randint(0, len(real_data) - 1)
                self.optimizer_co_recnet.zero_grad()
                noise = torch.randn(batch, 1, self.lr_h, self.lr_w).to(self.device)
                fake_lr, _ = self.lpm(self.co_G(noise), self.ref_image, fold_params = self.fold_params,
                                      divisor = self.divisor,
                                      n = self.lpm_iter,
                                      hard = self.lpm_hard, tao = tao)
                fake_hr = self.co_recnet(fake_lr)
                output = self.D_hr(fake_hr)
                errG = self.co_gen_beta * self.hr_generator_loss(output)
                if self.co_fd_beta > 0:
                    fd_loss = 0.01 * self.fdloss(fake_hr, self.hr_image)
                rec_loss = self.co_rec_beta * F.l1_loss(
                    self.co_recnet(noiseaug(real_data_lr[index])),
                    real_data[index])

                (errG + rec_loss + fd_loss).backward()
                self.optimizer_co_recnet.step()
                # self.co_recnet_EMA.update()
                self.co_G_EMA.update()

            EPOCHS.set_postfix(errD_fake = errD_fake_hr.item(), errD_real = errD_real_hr.item(), errG = errG.item(),
                               gradient_penalty = gradient_penalty.item(),
                               rec_loss = rec_loss.item(), w_loss = w_loss.item(), fd_loss = fd_loss.item())

            if (epoch + 1) % save_epoch == 0 or epoch == 0:
                self.co_G.eval()
                self.co_G_ema.eval()
                self._save_result(epoch, save_co = True)
                with torch.no_grad():
                    fake = self.G(_noise)
                    save_lr = fake
                    save_pm = \
                        self.lpm(save_lr, self.ref_image, fold_params = self.fold_params, divisor = self.divisor,
                                 n = self.lpm_iter, hard = False, tao = tao)[0]
                    save_hr1 = self.recnet(save_pm)

                    fake = self.co_G(_noise)
                    save_lr = fake
                    save_pm = \
                        self.lpm(save_lr, self.ref_image, fold_params = self.fold_params, divisor = self.divisor,
                                 n = self.lpm_iter, hard = False, tao = tao)[0]
                    save_lr = torch.cat([fake, save_pm], dim = 0)
                    functions.save_image(save_lr, f'{str(self.save_co_dir)}/lr_{epoch + 1}.png',
                                         nrow = fake.shape[0])
                    save_hr2 = self.co_recnet(save_pm)
                    save_hr = torch.cat([save_hr1, save_hr2], dim = 0)
                    functions.save_image(save_hr, f'{str(self.save_co_dir)}/hr_{epoch + 1}.png',
                                         nrow = fake.shape[0])

    def _save_result(self, epoch, save_recnet = False, save_lpm = False, save_gan = False, save_co = False):

        if save_recnet:
            torch.save(
                {'recnet': self.recnet.state_dict(),
                 'epoch': epoch + 1,
                 'optimizer_recnet': self.optimizer_recnet.state_dict(),
                 'lr_scheduler_recnet': self.lr_scheduler_recnet.state_dict()},
                '%s/recnet.pth' % str(self.save_weight_dir))
        if save_lpm:
            torch.save(
                {'lpm': self.lpm.state_dict(),
                 'optimizer_lpm': self.optimizer_lpm.state_dict()},
                '%s/lpm.pth' % str(self.save_weight_dir))
        if save_gan:
            torch.save({'G': self.G.state_dict(), 'G_ema': self.G_ema.state_dict() if self.use_ema else None,
                        'D_lr': self.D.state_dict(), 'optimizer_G': self.optimizer_G.state_dict(),
                        'optimizer_D': self.optimizer_D.state_dict(), 'epoch': epoch + 1,
                        'fixed_noise': self.fixed_noise},
                       f'{str(self.save_weight_dir)}/gan.pth')
        if save_co:
            torch.save({'co_G': self.co_G.state_dict(),
                        'co_G_ema': self.co_G_ema.state_dict(),
                        'co_recnet': self.co_recnet.state_dict(),
                        'co_recnet_ema': self.co_recnet_ema.state_dict(),
                        'D_hr': self.D_hr.state_dict(),
                        'fixed_noise': self.fixed_noise,
                        'optimizer_co_recnet': self.optimizer_co_recnet.state_dict(),
                        'optimizer_D_hr': self.optimizer_D_hr.state_dict(),
                        'epoch': epoch + 1},
                       f'{str(self.save_weight_dir)}/co_recnet.pth')

    def sample(self, patch_match_refine = True):
        if patch_match_refine is True:
            from Models.TTSR import TTSR
            kernel_size = 11
            refiner = TTSR().to(self.device)
            # ref = torch.cat([self.hr_image, torch.flip(self.hr_image, dims = [-1])], dim = -1)
            ref = torch.cat([self.hr_image], dim = -1)
            fold_params_refiner = {'kernel_size': (kernel_size, kernel_size), 'padding': kernel_size // 2, 'stride': 2,
                                   'dilation': 1}
            divisor_refiner = functions.getDivisor(self.hr_image, fold_params_refiner)

        which_G = self.co_G_ema if self.use_ema else self.co_G
        for i in range(36):
            z = torch.randn(1, 1, self.lr_h, self.lr_w).to(self.device)
            lr = which_G(z)
            lr_pm, _ = self.lpm(lr, self.ref_image, fold_params = self.fold_params, divisor = self.divisor,
                                n = self.lpm_iter, hard = False, tao = 0.00001)
            hr = self.co_recnet(lr_pm)
            functions.save_image(hr,
                                 '{}/{}.jpg'.format(str(self.save_final_syntheses_dir), 'hr' + str(i)),
                                 nrow = 1)

            if patch_match_refine is True:
                hr_refined, _ = refiner(hr, ref, ref,
                                        fold_params = fold_params_refiner,
                                        divisor = divisor_refiner, n = 10, lv = 1, skip = 4, return_img = True)

                functions.save_image(hr_refined,
                                     '{}/{}.jpg'.format(str(self.save_final_syntheses_dir), 'hr_refined' + str(i)),
                                     nrow = 1)

    def hr_learning(self):
        from Models.TTSR import TTSR
        ttsr = TTSR().to(self.divisor)
        self.save_hr_dir = self.save_dir / 'high_resolution'
        self.save_hr_dir.mkdir(parents = True, exist_ok = True)
        from Models.models import hrnet  # hrnet is an up-sampling network
        up_factor = opt.max_size // 256
        self.hrnet = hrnet(dim = 64, up_factor = up_factor).to(self.device)
        torchsummary.summary(self.hrnet, input_size = (3, self.hr_h, self.hr_w))

        hr_h = up_factor * self.hr_h
        hr_w = up_factor * self.hr_w
        hr = self.resize(self.origin, size = [hr_h, hr_w])

        _noise = torch.randn(1, 1, self.lr_h, self.lr_w).to(self.device)
        fake_lr, _ = self.lpm(self.co_G(_noise), self.ref_image, fold_params = self.fold_params,
                              divisor = self.divisor,
                              n = self.lpm_iter,
                              hard = self.lpm_hard)
        fake_hr = self.co_recnet(fake_lr)
        fake_probe = fake_hr
        ref = torch.cat([self.hr_image, torch.flip(self.hr_image, dims = [-1])], dim = -1)

        ks = 31
        fold_params = {'kernel_size': (ks, ks), 'padding': ks // 2, 'stride': 2, 'dilation': 1}
        divisor = functions.getDivisor(fake_probe, fold_params)
        fake_probe, _ = ttsr(fake_probe, ref, ref, fold_params = fold_params,
                             divisor = divisor, n = 10, lv = 1, skip = 4, return_img = True)

        functions.save_image(fake_hr, f'{str(self.save_hr_dir)}/fake.png',
                             nrow = 1)
        functions.save_image(fake_probe, f'{str(self.save_hr_dir)}/fake_hr.png',
                             nrow = 1)
        functions.save_image(hr, f'{str(self.save_hr_dir)}/hr.png',
                             nrow = 1)
        functions.save_image(self.hr_image, f'{str(self.save_hr_dir)}/lr.png',
                             nrow = 1)

        start_epoch = 0
        optimizer_hrnet = torch.optim.Adam(self.hrnet.parameters(), lr = 1e-4)
        try:
            pretrained_hrnet = torch.load(f'{str(self.save_weight_dir)}/hr.pth')
            self.hrnet.load_state_dict(pretrained_hrnet['hrnet'])
            start_epoch = pretrained_hrnet['epoch']
            optimizer_hrnet.load_state_dict(pretrained_hrnet['optimizer_G'])
            print('pretrained hrnet exists.')
        except:
            print('no pretrained hr net')
            pass

        end_epoch = self.opt.hr_epoch
        EPOCHS = tqdm(range(start_epoch, end_epoch))
        save_epoch = 100

        real_data = [hr, torch.flip(hr, dims = [-1])]
        real_data_lr = [self.hr_image, torch.flip(self.hr_image, dims = [-1])]
        real_hr = torch.cat(real_data)
        real_lr = torch.cat(real_data_lr)
        #############
        for epoch in EPOCHS:
            if self.opt.hr_batch == 2:
                index = [0, 1]
            else:
                index = [random.randint(0, 1)]

            optimizer_hrnet.zero_grad()
            rec = self.hrnet(real_lr[index, ...])
            rec_loss = F.mse_loss(rec, real_hr[index, ...])
            rec_loss.backward()
            optimizer_hrnet.step()

            EPOCHS.set_postfix(rec_loss = rec_loss.item())
            with torch.no_grad():
                if (epoch + 1) % save_epoch == 0 or epoch == 0:
                    self.co_G_ema.eval()
                    torch.save({'hrnet': self.hrnet.state_dict(),
                                'optimizer_G': optimizer_hrnet.state_dict(),
                                'epoch': epoch + 1},
                               f'{str(self.save_weight_dir)}/hr.pth')
                    # fake = self.G(_noise)

                    save_hr = self.hrnet(fake_probe)
                    functions.save_image(save_hr, f'{str(self.save_hr_dir)}/training_{epoch + 1}.png',
                                         nrow = 1)
                    functions.save_image(rec, f'{str(self.save_hr_dir)}/rec.png',
                                         nrow = 1)
                    del save_hr

        for i in range(36):
            noise = torch.randn(1, 1, self.lr_h, self.lr_w).to(self.device)
            with torch.no_grad():
                fake_lr, _ = self.lpm(self.co_G(noise), self.ref_image, fold_params = self.fold_params,
                                      divisor = self.divisor,
                                      n = self.lpm_iter,
                                      hard = self.lpm_hard)
                fake_hr_ = self.co_recnet(fake_lr)
                if i == 0:
                    divisor = functions.getDivisor(fake_hr_, fold_params)
                fake_hr_pm, _ = ttsr(fake_hr_, ref, ref, fold_params = fold_params,
                                     divisor = divisor, n = 10, lv = 1, skip = 4, return_img = True)
                fake_hr_pm = self.hrnet(fake_hr_pm)
                functions.save_image(fake_hr_pm, f'{str(self.save_hr_dir)}/hr_{i}.png',
                                     nrow = 1)
                del fake_hr_pm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ''' overall '''
    parser.add_argument('--gpu', type = int, default = 0)
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--hiddenprints', type = bool, default = True)
    parser.add_argument('--model_name', type = str, default = 'test')
    parser.add_argument('--input_dir', type = str, default = 'Input')
    parser.add_argument('--image_name', type = str, default = 'angkorwat.png')
    parser.add_argument('--save_dir', type = str, default = 'Result')
    parser.add_argument('--load_pretrained', type = str, default = 'True')
    parser.add_argument('--beta', type = float, default = 0.1)
    parser.add_argument('--downscale_factor', type = float, default = 8)
    parser.add_argument('--ref_model', type = str, default = 'test')
    parser.add_argument('--max_size', type = int, default = 256, help = 'an integer multiple of 256')
    parser.add_argument('--lpm_hard', type = bool, default = False)
    parser.add_argument('--use_flip', type = bool, default = True)

    # weight
    parser.add_argument('--co_weight_beta', type = float, default = 100)
    parser.add_argument('--co_gen_beta', type = float, default = 1)
    parser.add_argument('--co_rec_beta', type = float, default = 100)
    parser.add_argument('--co_fd_beta', type = float, default = 0)

    ''' model '''
    parser.add_argument('--recnet_dim', type = int, default = 64)
    parser.add_argument('--no_updown', type = bool, default = True)
    parser.add_argument('--code_dim', type = int, default = 3)
    parser.add_argument('--use_patch_dis', type = bool, default = False)
    parser.add_argument('--use_ema', type = bool, default = True)
    parser.add_argument('--upsample_mode', choices = ['conv', 'pixelshuffle', 'deconv', 'interp'], default = 'conv')
    parser.add_argument('--downsample_mode', choices = ['conv', 'pool', 'interp'], default = 'conv')
    parser.add_argument('--gen_ref', default = '', type = str)
    parser.add_argument('--updown_list', default = [1, 1, 1, 1, 1], type = int, nargs = '+')
    parser.add_argument('--lpm_iter', type = int, default = 10)
    parser.add_argument('--msd_stage', type = int, default = 3)
    parser.add_argument('--msd_dim', type = int, default = 64)

    ''' learning rate'''
    parser.add_argument('--lr_g', type = float, default = 1e-4)
    parser.add_argument('--lr_d', type = float, default = 1e-4)
    parser.add_argument('--lr_lpm', type = float, default = 1e-5)
    parser.add_argument('--lr_recnet', type = float, default = 5e-4)
    parser.add_argument('--lr_D_hr', type = float, default = 1e-4)
    parser.add_argument('--lr_rates', type = float, nargs = '+', default = [1e-6, 1e-4, 1e-4, 1e-4, 1e-4])

    ''' training epoch '''
    parser.add_argument('--gan_epoch', type = int, default = 5000)
    parser.add_argument('--D_iter', type = int, default = 1)
    parser.add_argument('--G_iter', type = int, default = 1)
    parser.add_argument('--gan_save_epoch', type = int, default = 200)

    parser.add_argument('--recnet_epoch', type = int, default = 5000)
    parser.add_argument('--recnet_save_epoch', type = int, default = 500)

    parser.add_argument('--co_G_iter', type = int, default = 1)
    parser.add_argument('--co_D_iter', type = int, default = 1)
    parser.add_argument('--co_epoch', type = int, default = 5000)
    parser.add_argument('--co_save_epoch', type = int, default = 100)

    parser.add_argument('--hr_epoch', type = int, default = 5000)

    # training batch
    parser.add_argument('--gan_batch', type = int, default = 8)
    parser.add_argument('--recnet_batch', type = int, default = 1)
    parser.add_argument('--co_batch', type = int, default = 1)
    parser.add_argument('--hr_batch', type = int, default = 2)

    # data augmentation
    parser.add_argument('--diffaug', type = str, help = 'color,translation,cutout', default = '')
    parser.add_argument('--mix_ratio', type = float, default = 0.3)

    opt = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{opt.gpu}'
    if opt.model_name is None:
        time_stamp = time.strftime('%Y-%m-%d#%H#%M#%S', time.localtime(time.time()))
        opt.model_name = '[' + time_stamp + ']'
    # opt.device = torch.device("cpu" if opt.gpu is None else "cuda:{}".format(opt.gpu))
    opt.device = torch.device("cpu" if opt.gpu is None else "cuda:{}".format(0))
    if torch.cuda.is_available() and opt.gpu is None:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    if type(opt.seed) is int:
        functions.seed_all(opt.seed)

    SinGAN = PetsGAN(opt)
    if opt.ref_model is not None:
        SinGAN._copy_weights(model_name = opt.ref_model, gan = True)
    SinGAN._init_all(gan = True, pm = True, recnet = True, co = True, load_gan = True, load_lpm = True,
                     load_recnet = True, load_co = True)
    SinGAN.external_learning()
    SinGAN.internal_learning()
    SinGAN.co_learning()
    SinGAN.sample()

    if opt.max_size > 256:
        if opt.max_size % 256 != 0:
            print("Please set max_size to an integer multiple of 256")
        SinGAN.hr_learning()
