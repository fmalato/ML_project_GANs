import torch
import numpy as np
import torch.nn as nn
import random

from utils import gram_matrix
from torch import Tensor
from torch.autograd import Variable


""" Baseline MSE Loss """
def LossE(device, image, target):
    criterion = nn.MSELoss().cuda()

    return criterion(image, target.to(device))


""" Perceptual Loss """
def LossP(vgg, device, image, target):
    criterion = nn.MSELoss()
    criterion.cuda()
    vgg_2 = vgg[0].float()
    vgg_5 = vgg[1].float()
    vgg_2.cuda()
    vgg_5.cuda()

    extr_feat = vgg_2(image.float())
    real_feat = vgg_2(target.float())
    loss_2 = criterion(extr_feat, real_feat.detach())
    extr_feat = vgg_5(image.float())
    real_feat = vgg_5(target.float())
    loss_5 = criterion(extr_feat, real_feat.detach())

    return (2e-1)*loss_2 + (2e-2)*loss_5


""" GAN generator and discriminator Losses """
def LossA(generator, discriminator, device, image, target, bicub, optim_d, PCM, lossT=False):
    disc_train_real = target.float().to(device)
    batch_size = disc_train_real.size(0)

    if lossT:
        criterion = nn.BCELoss(weight=torch.full((batch_size,), 2, device=device))
    else:
        criterion = nn.BCELoss(weight=torch.full((batch_size,), 1, device=device))
    # Discriminator true
    optim_d.zero_grad()
    label = torch.full((batch_size,), random.uniform(0.9, 1.0), device=device).cuda()
    output_d = discriminator(disc_train_real.float()).view(-1)
    loss_d_real = criterion(output_d.float(), label.float()).cuda()
    D_x = output_d.mean().item()
    loss_d_real.backward()

    # Discriminator false
    output_g = generator(image.float(), bicub.float())
    output_d = discriminator(torch.add(torch.add(output_g.detach(), PCM), bicub).clamp(0, 255)).view(-1)
    label.fill_(random.uniform(0.0, 0.1))
    loss_d_fake = criterion(output_d.float(), label.float()).cuda()
    D_G_z1 = output_d.mean().item()
    loss_d = loss_d_real + loss_d_fake

    loss_d_fake.backward()
    optim_d.step()

    # Generator
    label.fill_(random.uniform(0.9, 1.0))
    output_d = discriminator(output_g.float()).view(-1)
    loss_g = criterion(output_d.float(), label.float()).cuda()
    D_G_z2 = output_d.mean().item()

    return loss_g, loss_d, D_x, D_G_z1, D_G_z2


""" Texture Loss """
def LossT(vgg, device, image, target, patch_size=16):
    criterion = nn.MSELoss()

    vgg_1 = vgg[0]
    vgg_2 = vgg[1]
    vgg_3 = vgg[2]
    image = torch.split(image, 1, dim=0)
    img_size = image[0].shape[2]
    batch_size = int(img_size / patch_size)**2
    patches = []
    for el in image:
        new = el.unfold(1, 3, 3).unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        new = new.reshape((batch_size, 3, patch_size, patch_size))
        patches.append(new)
    patches = torch.cat(patches)
    patches.to(device)
    target = torch.split(target, 1, dim=0)
    img_size = target[0].shape[2]
    batch_size = int(img_size / patch_size) ** 2
    patches_target = []
    for el in target:
        new = el.unfold(1, 3, 3).unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        new = new.reshape((batch_size, 3, patch_size, patch_size))
        patches_target.append(new)
    patches_target = torch.cat(patches_target)
    patches_target.to(device)

    loss_1 = criterion(gram_matrix(vgg_1(patches.float())).float(), gram_matrix(vgg_1(patches_target.float())).float())
    loss_2 = criterion(gram_matrix(vgg_2(patches.float())).float(), gram_matrix(vgg_2(patches_target.float())).float())
    loss_3 = criterion(gram_matrix(vgg_3(patches.float())).float(), gram_matrix(vgg_3(patches_target.float())).float())

    return (3e-7) * loss_1 + (1e-6) * loss_2 + (1e-6) * loss_3



