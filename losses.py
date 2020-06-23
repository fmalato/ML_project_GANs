import torch
import torch.nn as nn
import random
import numpy as np

from utils import gram_matrix, true_or_false
from torch.autograd import Variable
from torch import Tensor


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
    l_true = Variable(torch.full((batch_size,), 0.9, device=device), requires_grad=False)
    l_fake = Variable(torch.full((batch_size,), 0.0, device=device), requires_grad=False)
    l_true_g = Variable(torch.full((batch_size,), 1.0, device=device), requires_grad=False)

    # Discriminator
    optim_d.zero_grad()
    output_t = discriminator(target).view(-1).clamp(1e-7, 1-1e-7)
    output_d = discriminator(output_g.detach()).view(-1).clamp(1e-7, 1 - 1e-7)
    d_x = output_t.mean().item()

    if train_disc:
        discriminator.allow_train()
    # loss_d = Variable(- 0.5 * torch.mean(torch.log(output_t)) - 0.5 * torch.mean(torch.log(l_true_g - output_d)), requires_grad=True)
    loss_d = Variable(Tensor(criterion(output_d, l_fake) + criterion(output_t, l_true)), requires_grad=True)
    if train_disc:
        loss_d.backward()
        optim_d.step()
    discriminator.deny_train()

    # Generator
    d_g_z = output_d.mean().item()
    #loss_g = Variable(- 0.5 * torch.mean(torch.log(output_d.detach())), requires_grad=True)
    loss_g = Variable(Tensor(criterion(output_d, l_true_g)), requires_grad=True)
    if lossT:
        loss_g = loss_g * 2
        loss_d = loss_d * 2

    return loss_g, loss_d, d_x, d_g_z


""" Texture Loss """
def LossT(device, extr_feat, true_feat):

    return torch.mean((gram_matrix(extr_feat) - gram_matrix(true_feat)) ** 2)



