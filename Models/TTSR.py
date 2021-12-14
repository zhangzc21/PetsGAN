import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign = -1):
        super(MeanShift, self).__init__(3, 3, kernel_size = 1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        # self.requires_grad = False
        self.weight.requires_grad = False
        self.bias.requires_grad = False

class LTE(torch.nn.Module):
    def __init__(self, requires_grad = True, rgb_range = 1):
        super(LTE, self).__init__()

        ### use vgg19 weights to initialize
        vgg_pretrained_features = models.vgg19(pretrained = True).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.slice1.parameters():
                param.requires_grad = requires_grad
            for param in self.slice2.parameters():
                param.requires_grad = requires_grad
            for param in self.slice3.parameters():
                param.requires_grad = requires_grad

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.slice1(x)
        x_lv1 = x
        x = self.slice2(x)
        x_lv2 = x
        x = self.slice3(x)
        x_lv3 = x
        return x_lv1, x_lv2, x_lv3

class TTSR(nn.Module):
    def __init__(self, requires_grad = False):
        super(TTSR, self).__init__()
        self.LTE = LTE(requires_grad = requires_grad)
        self.SearchTransfer = SearchTransfer()

    def forward(self, lrsr, ref, refsr,
                fold_params = {'kernel_size': (3, 3), 'padding': 1, 'stride': 1, 'dilation': 1}, divisor = None, n = 1, lv = 1, skip = 1, return_img = False):
        # if lr is not None:
        #     lrsr = torch.nn.functional.interpolate(lr, size = ref.shape[-2:], mode = 'bilinear', align_corners = True)
        ref = ref.repeat([lrsr.shape[0], 1, 1, 1])
        refsr = refsr.repeat([lrsr.shape[0], 1, 1, 1])
        refsr_lv1, refsr_lv2, refsr_lv3 = self.LTE((refsr.detach() + 1.) / 2.)
        ref_lv1, _, _ = self.LTE((ref.detach() + 1.) / 2.)
        if lv == 0:
            refsr_f = (refsr.detach() + 1.) / 2.
        if lv == 1:
            refsr_f = refsr_lv1[:,::skip,...]
        if lv == 2:
            refsr_f = refsr_lv2[:,::skip,...]
        if lv == 3:
            refsr_f = refsr_lv3[:,::skip,...]

        T_org = lrsr
        for _ in range(n):
            lrsr_lv1, lrsr_lv2, lrsr_lv3 = self.LTE((T_org.detach() + 1.) / 2.)
            if lv == 0:
                lrsr_f = (T_org.detach() + 1.) / 2.
            if lv == 1:
                lrsr_f = lrsr_lv1[:,::skip,...]
            if lv == 2:
                lrsr_f = lrsr_lv2[:,::skip,...]
            if lv == 3:
                lrsr_f = lrsr_lv3[:,::skip,...]

            # S, T_lv3, T_lv2, T_lv1 = self.SearchTransfer(lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3)
            if return_img is True:
                ref_f = (ref + 1)/2
            else:
                ref_f = ref_lv1
            T_org, S = self.SearchTransfer(lrsr_f, refsr_f, ref_f, fold_params, lv = lv)
            if divisor is not None:
                T_org *= divisor.expand(T_org.shape)
                T_org = (T_org-0.5)*2
        return T_org , S


class SearchTransfer(nn.Module):
    def __init__(self):
        super(SearchTransfer, self).__init__()

    def bis(self, input, dim, index):
        # batch index select
        # input: [N, ?, ?, ...]
        # dim: scalar > 0
        # index: [N, idx]

        views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, lrsr_lv1, refsr_lv1, org, fold_params, lv):
        ### search
        lrsr_lv1_unfold = F.unfold(lrsr_lv1, **fold_params)
        refsr_lv1_unfold = F.unfold(refsr_lv1, **fold_params)
        refsr_lv1_unfold = refsr_lv1_unfold.permute(0, 2, 1)  # (1, HxW, Cx3x3)

        refsr_lv1_unfold = F.normalize(refsr_lv1_unfold, dim = 2)  # [N, Hr*Wr, C*k*k]
        lrsr_lv1_unfold = F.normalize(lrsr_lv1_unfold, dim = 1)  # [N, C*k*k, H*W]

        R_lv1 = refsr_lv1_unfold @ lrsr_lv1_unfold  # [N, Hr*Wr, H*W]
        R_lv1_star, R_lv1_star_arg = torch.max(R_lv1, dim = 1)  # [N, H*W]

        ### transfer
        # ref_lv3_unfold = F.unfold(ref_lv3, kernel_size=(3, 3), padding=1)
        # ref_lv2_unfold = F.unfold(ref_lv2, kernel_size=(6, 6), padding=2, stride=2)
        # 将 ref 按块提取
        # org_unfold =

        # T_lv3_unfold = self.bis(ref_lv3_unfold, 2, R_lv3_star_arg)
        # T_lv2_unfold = self.bis(ref_lv2_unfold, 2, R_lv3_star_arg)
        # T_lv1_unfold = self.bis(ref_lv1_unfold, 2, R_lv1_star_arg)

        # T_lv3 = F.fold(T_lv3_unfold, output_size=lrsr_lv3.size()[-2:], kernel_size=(3,3), padding=1) / (3.*3.)
        # T_lv2 = F.fold(T_lv2_unfold, output_size=(lrsr_lv3.size(2)*2, lrsr_lv3.size(3)*2), kernel_size=(6,6), padding=2, stride=2) / (3.*3.)
        # T_lv1 = F.fold(T_lv1_unfold, output_size = (lrsr_lv1.size(2), lrsr_lv1.size(3)), kernel_size = self.kerner_size,
        #                padding = self.padding) / (3. * 3.)

        # S = R_lv1_star_arg.view(R_lv1_star.size(0), 1, lrsr_lv1.size(2), lrsr_lv1.size(3))
        S = R_lv1_star_arg
        lv_ = max(lv-1, 0)
        fp = {'kernel_size': (2**(lv_) * fold_params['kernel_size'][0], 2**(lv_) * fold_params['kernel_size'][1]),
              'padding': 2**(lv_) * fold_params['padding'], 'stride': 2**(lv_) * fold_params['stride']}
        fold_params = fp

        org_unfold = F.unfold(org, **fold_params)
        T_org_unfold = self.bis(org_unfold, 2, R_lv1_star_arg)
        T_org = F.fold(T_org_unfold, output_size = (lrsr_lv1.size(2) * 2**(lv_), lrsr_lv1.size(3) * 2**(lv_)), **fold_params)

        # org_unfold = F.unfold(refsr_lv1, **fold_params)
        # T_org_unfold = self.bis(org_unfold, 2, R_lv1_star_arg)
        # T_org = F.fold(T_org_unfold, output_size = (lrsr_lv1.size(2) * 2**(lv-1), lrsr_lv1.size(3) * 2**(lv-1)), **fold_params)

        return T_org, S
