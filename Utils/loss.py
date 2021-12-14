# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F


def DiscriminatoryLoss(D, x, y, class_, ftr_num=4):
    with torch.no_grad():
        real_output, real_features = D(x, class_)
    fake_output, fake_features = D(y, class_)
    loss = 0
    for i in range(ftr_num):
        f_id = -i - 1
        loss = loss + F.l1_loss(fake_features[f_id], real_features[f_id])
    return loss, fake_output
