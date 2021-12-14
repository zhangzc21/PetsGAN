import torch
import torchvision.transforms as T
import random
from Utils.tps import tps_warp
from PIL import Image
import numpy as np
import torch.nn.functional as F
import functools
import Utils.functions as functions
import torchvision
import matplotlib.pyplot as plt
def show_image(x):
    x = torchvision.utils.make_grid(x, normalize = True)
    x = x.detach().cpu().permute(1, 2, 0).numpy()
    plt.imshow(x)
    plt.show()


resize = functools.partial(F.interpolate, mode = 'bicubic', align_corners = True)


def tps(x):
    x = x.cpu().detach().squeeze(0).permute(1, 2, 0).numpy()
    image_t = tps_warp(x, points_per_dim = 4, scale = 0.05)
    # img = np.concatenate((x, image_t), axis = 1)
    return torch.from_numpy(image_t).permute(2, 0, 1).unsqueeze(0)

class DataTrans():
    def __init__(self):
        self.CJ = T.ColorJitter(brightness = 0.1, contrast = 0.1, saturation = 0.1, hue = 0.1)
        # RF = T.RandomAffine(30, translate = None, scale = None, shear = None, fillcolor = None)
        RHF = T.RandomHorizontalFlip(p = 0.5)
        # RR = T.RandomRotation(30)
        # self.GB = T.GaussianBlur(kernel_size = 3)
        TPS = tps
        self.full = [RHF, TPS]
        self.resize = functools.partial(F.interpolate, mode = 'bicubic', align_corners = True)
    def gen(self, x, mode = 'full'):
        random.shuffle(self.full)
        ts = self.full[:random.randint(1, len(self.full))]
        for t in ts:
            x = t(x)
        # if mode == 'sub':
        #     H, W = x.shape[-2:]
        #     min_H, min_W = H // 8, W // 8
        #     sH = random.randint(min_H, H // 2)
        #     sW = random.randint(min_W, W // 2)
        #     rH = random.randint(0, H - 1 - sH)
        #     rW = random.randint(0, W - 1 - sH)
        #     x = x[..., rH: (rH + sH), rW: (rW + sW)]

        # label, shapes = functions.create_pyramid(x, self.opt)
        # data = label[0]
        # if random.random() < 0.3:
        #     data = self.GB(data)
        # if random.random() > 0.9:
        #     data = self.CJ(data)
        # if random.random() < 0.3:
        #     data = data + 0.1 * torch.randn_like(data)
        return x

class DataCrop():
    def __init__(self, opt):
        self.opt = opt
        self.CJ = T.ColorJitter(brightness = 0.1, contrast = 0.1, saturation = 0.1, hue = 0.1)
        # RF = T.RandomAffine(30, translate = None, scale = None, shear = None, fillcolor = None)
        RHF = T.RandomHorizontalFlip(p = 0.5)
        # RR = T.RandomRotation(30)
        self.GB = T.GaussianBlur(kernel_size = 3)
        self.identity = lambda x : x
        TPS = tps
        self.full = [self.identity, RHF, TPS]
        self.resize = functools.partial(F.interpolate, mode = 'bicubic', align_corners = True)
        self.shapes = opt.shapes
    def gen(self, image,  num = 50, batch = 100):
        data = resize(image, self.shapes[0])
        for i in range(num):
            x = image
            random.shuffle(self.full)
            ts = self.full[:random.randint(1, len(self.full))]
            for t in ts:
                x = t(x)
            H, W = x.shape[-2:]
            min_H, min_W = H // 5, W // 5
            for j in range(batch):
                sH = random.randint(min_H, H)
                sW = random.randint(min_W, W)
                if sH < sW:
                    sW = int(W / H *sH)
                else:
                    sH = int(H / W * sW)
                rH = random.randint(0, H - sH)
                rW = random.randint(0, W - sW)
                data_ = x[..., rH: (rH + sH), rW: (rW + sW)]

                data_ = resize(data_, self.shapes[0])
                data = torch.cat([data, data_.to(self.opt.device)], dim=0)
            # if random.random() < 0.3:
            #     data = self.GB(data)
            # if random.random() > 0.9:
            #     data = self.CJ(data)
            # if random.random() < 0.3:
            #     data = data + 0.1 * torch.randn_like(data)
        return data

def getDivisor(x, fp):
    img = torch.ones_like(x)
    unfold_img = F.unfold(img, **fp)
    fold_img = F.fold(unfold_img, output_size = x.shape[-2:], **fp)
    divisor =  1 / fold_img
    divisor[fold_img == 0] = 0
    return divisor


def DataPrepare(x, opt, batch = 20):
    Trans = DataTrans(opt)
    DataFull_HR = []
    DataFull_LR = []

    different_scales, shapes = functions.create_pyramid(x, opt)

    DataShuffle_HR = []
    DataShuffle_LR = []

    DataCrop_HR = {}
    DataCrop_LR = {}
    # 准备 变换的图像
    for i in range(batch):
        data, label, _ = Trans.gen(x, mode = 'full')
        DataFull_HR.append(label)
        DataFull_LR.append(data)

    for i in [8]:
        k1, k2 = x.shape[-2] // i, x.shape[-1] // i
        k1, k2 = 8, 8
        s1, s2 = k1, k2
        unfold_x = F.unfold(x, kernel_size = (k1, k2), stride = (s1, s2), padding = 0)

        crops = torch.cat([unfold_x[:,:k1*k2, :],unfold_x[:,k1*k2:2*k1*k2, :], unfold_x[:,2*k1*k2:3*k1*k2, :]])
        crops = crops.permute(2, 0 ,1)
        crops = crops.view(-1,3,k1,k2)

        crops_lr, shapes = functions.create_pyramid(crops, opt)
        DataCrop_HR[i] = crops_lr
        DataCrop_LR[i] = (crops_lr[0], shapes)
        # 准备Crop的图像
    # for _ in range(batch * 4):
    #     n = len(DataFull_LR)
    #     i = random.randint(0, n - 1)
    #     x = DataFull_HR[i]
    #     H, W = x.shape[-2:]
    #     min_H, min_W = H // 5, W // 5
    #     sH = random.randint(min_H, H // 2)
    #     sW = random.randint(min_W, W // 2)
    #     rH = random.randint(0, H - 1 - sH)
    #     rW = random.randint(0, W - 1 - sH)
    #     HR = x[..., rH: (rH + sH), rW: (rW + sW)]
    #     LR = resize(HR, different_scales[0].shape[-2:])
    #     if random.random() < 0.3:
    #         LR = Trans.GB(LR)
    #     if random.random() > 0.9:
    #         LR = Trans.CJ(LR)
    #     if random.random() < 0.3:
    #         LR = LR + 0.1 * torch.randn_like(LR)
    #     DataCrop_HR.append(HR)
    #     DataCrop_LR.append(LR)

    # 准备 shuffle的图像
    # for i in range(batch // 2):
    #     r = random.choices([2,4,8])[0]
    #
    #     img = different_scales[-1]
    #     shape = img.shape[-2:]
    #     fp = dict(kernel_size = (shape[0] // r, shape[1] // r), padding = 0, stride = (shape[0] // r, shape[1] // r),
    #               dilation = 1)
    #     divisor = getDivisor(img, fp)
    #     unfold_img = F.unfold(img, **fp)
    #     index = torch.randperm(unfold_img.shape[-1])
    #     img = F.fold(unfold_img[..., index], output_size = shape, **fp) * divisor
    #     DataFull_HR.append(img)
    #
    #     img = different_scales[0]
    #     shape = img.shape[-2:]
    #     fp = dict(kernel_size = (shape[0] // r, shape[1] // r), padding = 0, stride = (shape[0] // r, shape[1] // r),
    #               dilation = 1)
    #     divisor = getDivisor(img, fp)
    #     unfold_img = F.unfold(img, **fp)
    #     img = F.fold(unfold_img[..., index], output_size = shape, **fp) * divisor
    #     DataFull_LR.append(img)

    # for i in range(batch // 2):
    #     r = random.randint(2, 8)
    #
    #     img = different_scales[-1]
    #     shape = img.shape[-2:]
    #     h = shape[0] // r
    #     w = shape[1] // r
    #     s1 = random.randint(3, h // 2)
    #     s2 = random.randint(3, w // 2)
    #     fp = dict(kernel_size = (h, w), padding = 0, stride = (s1, s2), dilation = 1)
    #     divisor = getDivisor(img, fp)
    #     unfold_img = F.unfold(img, **fp)
    #     index = torch.randperm(unfold_img.shape[-1])
    #     img = F.fold(unfold_img[..., index], output_size = shape, **fp) * divisor
    #     DataFull_HR.append(img)
    #     DataFull_LR.append(resize(img, shapes[0]))

    return DataFull_HR, DataFull_LR, DataCrop_HR, DataCrop_LR