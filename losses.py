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
def LossA(generator, discriminator, device, image, target, optim_d, lossT=False):

    criterion = nn.BCELoss()
    train_d = False
    # Discriminator
    optim_d.zero_grad()
    disc_train_real = target.to(device)
    batch_size = disc_train_real.size(0)
    label = torch.full((batch_size,), 1, device=device).cuda()
    output_d = discriminator(disc_train_real).view(-1)
    loss_d_real = criterion(output_d, label).cuda()
    if lossT:
        loss_d_real *= 2
    if loss_d_real.item() > 0.3:
        loss_d_real.backward()
        train_d = True

    output_g = generator(image)
    output_d = discriminator(output_g.detach())
    label.fill_(0)
    loss_d_fake = criterion(output_d, label).cuda()
    if lossT:
        loss_d_fake *= 2
    loss_d = loss_d_real + loss_d_fake
    if loss_d_fake.item() > 0.3:
        loss_d_fake.backward()
        train_d = True
    if train_d:
        optim_d.step()

    # Generator
    label.fill_(1)
    output_d = discriminator(output_g).view(-1)
    loss_g = criterion(output_d, label).cuda()
    if lossT:
        loss_g *= 2

    return loss_g, loss_d


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



