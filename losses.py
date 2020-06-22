import torch
import torch.nn as nn
import random
import numpy as np

from utils import gram_matrix, true_or_false


""" Baseline MSE Loss """
def LossE(device, image, target):
    criterion = nn.MSELoss().cuda()

    return criterion(image, target.to(device)).cuda()


""" Perceptual Loss """
def LossP(vgg, device, image, target):

    vgg_2 = vgg[0].float()
    vgg_5 = vgg[1].float()

    extr_feat = vgg_2(image)
    real_feat = vgg_2(target)
    loss_2 = torch.mean((extr_feat - real_feat) ** 2)
    extr_feat = vgg_5(image)
    real_feat = vgg_5(target)
    loss_5 = torch.mean((extr_feat - real_feat) ** 2)

    return (2e-1*loss_2 + 2e-2*loss_5).cuda()


""" GAN generator and discriminator Losses """
def LossA(discriminator, device, output_g, target, optim_d, lossT=False, train_disc=True):
    batch_size = output_g.size(0)
    criterion = nn.MSELoss()
    l_true = torch.full((batch_size,), 1.0, device=device)
    l_fake = torch.full((batch_size,), 0.0, device=device)

    # Discriminator
    optim_d.zero_grad()
    output_t = discriminator(target).view(-1).clamp(1e-7, 1-1e-7)
    output_d = discriminator(output_g.detach()).view(-1).clamp(1e-7, 1 - 1e-7)
    d_x = output_t.mean().item()
    #loss_d = - 1.0 * torch.log(output_t) - 1.0 * torch.log(torch.full((batch_size,), 1., device=device) - output_d)
    loss_d = criterion(output_d, l_fake) + criterion(output_t, l_true)
    if train_disc:
        if lossT:
            loss_d *= 2
        loss_d.mean().backward()

        optim_d.step()

    # Generator
    d_g_z = output_d.mean().item()
    #loss_g = - 1.0 * torch.log(output_d)
    loss_g = criterion(output_d, l_true)
    if lossT:
        loss_g *= 2

    return loss_g.cuda(), loss_d.cuda(), d_x, d_g_z


""" Texture Loss """
def LossT(vgg, device, image, target, patch_size=16):
    # Images
    image = torch.split(image, 1, dim=0)
    target = torch.split(target, 1, dim=0)

    vgg_1 = vgg[0].cuda()
    vgg_2 = vgg[1].cuda()
    vgg_3 = vgg[2].cuda()
    
    patches = []
    patches_target = []
    batch_size = int(image[0].shape[2] / patch_size) ** 2
    idx = 0
    for el in image:
        # one patch every 16 is computed in order to reduce computation. On the dataset there are 655114*64 16x16 patches
        # so I guess a sixteenth (kind of 3kk per epoch) of them is a good trade-off for a 15h speed up on the training
        if idx % 16 == 0:
            new = el.unfold(1, 3, 3).unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
            new = new.reshape((batch_size, 3, patch_size, patch_size))
            new = torch.split(new, 1, dim=0)
            patches += new
        idx += 1
    del image
    idx = 0
    for el in target:
        if idx % 16 == 0:
            new = el.unfold(1, 3, 3).unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
            new = new.reshape((batch_size, 3, patch_size, patch_size))
            new = torch.split(new, 1, dim=0)
            patches_target += new
    del target
    del idx

    loss_1 = torch.Tensor(np.zeros(1)).cuda()
    loss_2 = torch.Tensor(np.zeros(1)).cuda()
    loss_3 = torch.Tensor(np.zeros(1)).cuda()

    for i, t in zip(patches, patches_target):
        loss_1 += torch.mean((gram_matrix(vgg_1(i)) - gram_matrix(vgg_1(t))) ** 2)
        loss_2 += torch.mean((gram_matrix(vgg_2(i)) - gram_matrix(vgg_2(t))) ** 2)
        loss_3 += torch.mean((gram_matrix(vgg_3(i)) - gram_matrix(vgg_3(t))) ** 2)

    loss_1 = torch.mean(loss_1)
    loss_2 = torch.mean(loss_2)
    loss_3 = torch.mean(loss_3)

    return (3e-7 * loss_1 + 1e-6 * loss_2 + 1e-6 * loss_3).cuda()



