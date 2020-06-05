import torch
import numpy as np
import torch.nn as nn

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
    vgg_2 = vgg[0]
    vgg_5 = vgg[1]
    vgg_2.cuda()
    vgg_5.cuda()

    extr_feat = vgg_2(image)
    real_feat = vgg_2(target)
    loss_2 = criterion(extr_feat, real_feat.detach())
    extr_feat = vgg_5(image)
    real_feat = vgg_5(target)
    loss_5 = criterion(extr_feat, real_feat.detach())

    return (2e-1)*loss_2 + (2e-2)*loss_5


""" GAN generator and discriminator Losses """
def LossA(generator, discriminator, device, image, target, bicub, optim_d, lossT=False):
    disc_train_real = target.to(device)
    batch_size = disc_train_real.size(0)
    #train_d = False
    if lossT:
        criterion = nn.BCELoss(weight=torch.full((batch_size,), 2, device=device))
    else:
        criterion = nn.BCELoss(weight=torch.full((batch_size,), 1, device=device))
    # Discriminator true
    optim_d.zero_grad()
    label = torch.full((batch_size,), 1, device=device).cuda()
    output_d = discriminator(disc_train_real).view(-1)
    #good = len([1 for x in output_d.tolist() if x >= 0.5])
    #perf_1 = good / batch_size
    loss_d_real = criterion(output_d, label).cuda()
    D_x = output_d.mean().item()
    """if perf_1 < 0.9:
        loss_d_real.backward()
        train_d = True"""
    loss_d_real.backward()
    # Discriminator false
    output_g = generator(image, bicub)
    output_d = discriminator(output_g.detach()).view(-1)
    #good = len([0 for x in output_d.tolist() if x < 0.5])
    #perf_0 = good / batch_size
    label.fill_(0)
    loss_d_fake = criterion(output_d, label).cuda()
    D_G_z1 = output_d.mean().item()
    loss_d = loss_d_real + loss_d_fake
    """if perf_0 < 0.9:
        loss_d_fake.backward()
        train_d = True
    if train_d:
        optim_d.step()"""
    loss_d_fake.backward()
    optim_d.step()

    # Generator
    label.fill_(1)
    output_d = discriminator(output_g).view(-1)
    loss_g = criterion(output_d, label).cuda()
    D_G_z2 = output_d.mean().item()

    return loss_g, loss_d, D_x, D_G_z1, D_G_z2


""" Texture Loss """
def LossT(vgg, device, image, target):
    criterion = nn.MSELoss()
    criterion.cuda()
    loss = 0.0
    vgg_1 = vgg[0]
    vgg_2 = vgg[1]
    vgg_3 = vgg[2]
    vgg_1.cuda()
    vgg_2.cuda()
    vgg_3.cuda()
    image = image.view((3, image.shape[2], image.shape[3]))
    target = target.view((3, target.shape[2], target.shape[3]))
    patches = image.data.unfold(0, 3, 3).unfold(1, 16, 16).unfold(2, 16, 16)
    patches_target = target.data.unfold(0, 3, 3).unfold(1, 16, 16).unfold(2, 16, 16)
    batch_size = patches.shape[1] * patches.shape[2]
    patches = patches.reshape((batch_size, 3, 16, 16))
    patches_target = patches_target.reshape((batch_size, 3, 16, 16))
    pat_list = torch.split(patches, batch_size)
    pat_list = pat_list[0]
    pat_tar_list = torch.split(patches_target, batch_size)
    pat_tar_list = pat_tar_list[0]
    loss_1 = loss_2 = loss_3 = 0.0
    for i in range(batch_size / 16):
        loss_1 += criterion(gram_matrix(vgg_1(pat_list[i].view((1, 3, 16, 16)))),
                            gram_matrix(vgg_1(pat_tar_list[i].view((1, 3, 16, 16)))))
        loss_2 += criterion(gram_matrix(vgg_2(pat_list[i].view((1, 3, 16, 16)))),
                            gram_matrix(vgg_2(pat_tar_list[i].view((1, 3, 16, 16)))))
        loss_3 += criterion(gram_matrix(vgg_3(pat_list[i].view((1, 3, 16, 16)))),
                            gram_matrix(vgg_3(pat_tar_list[i].view((1, 3, 16, 16)))))

    return (3e-7) * torch.div(loss_1, batch_size) + (1e-6) * torch.div(loss_2, batch_size) + (1e-6) * torch.div(loss_3, batch_size)



