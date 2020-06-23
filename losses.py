import torch
import torch.nn as nn
import random
import numpy as np

from utils import gram_matrix, true_or_false
from torch.autograd import Variable


""" Baseline MSE Loss """
def LossE(device, image, target):
    criterion = nn.MSELoss().cuda()

    return criterion(image, target.to(device)).cuda()


""" Perceptual Loss """
def LossP(device, extr_feat_2, true_feat_2, extr_feat_5, true_feat_5):

    loss_2 = torch.mean((extr_feat_2 - true_feat_2) ** 2)
    loss_5 = torch.mean((extr_feat_5 - true_feat_5) ** 2)

    # PA and PAT
    #return 2e-1*loss_2 + 2e-2*loss_5
    # P
    return 2e-1 * loss_2 + \
           loss_5


""" GAN generator and discriminator Losses """
def LossA(discriminator, device, output_g, target, optim_d, lossT=False, train_disc=True):
    batch_size = output_g.size(0)
    criterion = nn.BCELoss()
    l_true = torch.full((batch_size,), 0.9, device=device)
    l_fake = torch.full((batch_size,), 0.0, device=device)
    l_true_g = torch.full((batch_size,), 1.0, device=device)

    # Discriminator
    optim_d.zero_grad()
    output_t = discriminator(target).view(-1).clamp(1e-7, 1-1e-7)
    output_d = discriminator(output_g.detach()).view(-1).clamp(1e-7, 1 - 1e-7)
    d_x = output_t.mean().item()
    loss_d = Variable(- 0.5 * torch.mean(torch.log(output_t)) - 0.5 * torch.mean(torch.log(l_true_g - output_d)), requires_grad=True)

    #loss_d = criterion(output_d, l_fake) + criterion(output_t, l_true)
    if train_disc:
        discriminator.allow_train()
        loss_d.mean().backward()
        optim_d.step()
    discriminator.deny_train()

    # Generator
    d_g_z = output_d.mean().item()
    loss_g = Variable(- 0.5 * torch.mean(torch.log(output_d.detach())), requires_grad=True)
    #loss_g = criterion(output_d, l_true_g)
    if lossT:
        loss_g *= 2
        loss_d *= 2

    return loss_g.cuda(), loss_d.cuda(), d_x, d_g_z


""" Texture Loss """
def LossT(device, extr_feat, true_feat):

    return torch.mean((gram_matrix(extr_feat) - gram_matrix(true_feat)) ** 2)



