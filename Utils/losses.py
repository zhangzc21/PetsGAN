import torch
import torch.nn.functional as F



def loss_gan_dis(dis_fake, dis_real):
  L1 = -torch.mean(torch.log(torch.sigmoid(dis_real)))
  L2 = -torch.mean(torch.log(1-torch.sigmoid(dis_fake)))
  return L1, L2


def loss_gan_gen(dis_fake):
  loss = torch.mean(torch.log(1-torch.sigmoid(dis_fake)))
  return loss

def loss_dcgan_dis(dis_fake, dis_real):
  L1 = torch.mean(F.softplus(-dis_real))
  L2 = torch.mean(F.softplus(dis_fake))
  return L1, L2


def loss_dcgan_gen(dis_fake):
  loss = torch.mean(F.softplus(-dis_fake))
  return loss

def loss_ls_dis(dis_fake, dis_real):
  L1 = torch.mean((1-dis_real)**2)
  L2 = torch.mean(dis_fake**2)
  return L1, L2


def loss_ls_gen(dis_fake):
  loss = torch.mean((1-dis_fake)**2)
  return loss


# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real):
  loss_real = torch.mean(F.relu(1. - dis_real))
  loss_fake = torch.mean(F.relu(1. + dis_fake))
  return loss_real, loss_fake


def loss_hinge_gen(dis_fake):
  loss = -torch.mean(dis_fake)
  return loss


def loss_wgan_dis(dis_fake, dis_real):
  loss_real = -torch.mean(dis_real)
  loss_fake = torch.mean( dis_fake)
  return loss_real, loss_fake


def loss_wgan_gen(dis_fake):
  loss = -torch.mean(dis_fake)
  return loss

def loss_spa(x):
  return -torch.sqrt((x ** 2).sum(dim = -1)).mean()

def loss_spa2(x):
  return torch.nn.functional.l1_loss(x, torch.eye(x.shape[-1]).squeeze(0).expand(*x.shape).to(x.device))