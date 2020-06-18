import torch
import torch.nn as nn
import random
import numpy as np

from utils import gram_matrix


""" Baseline MSE Loss """
def LossE(device, image, target):
    criterion = nn.MSELoss().cuda()

    return criterion(image, target.to(device)).cuda()


""" Perceptual Loss """
def LossP(vgg, device, image, target):

    vgg_2 = vgg[0].float()
    vgg_5 = vgg[1].float()
    vgg_2.cuda()
    vgg_5.cuda()

    extr_feat = vgg_2(image)
    real_feat = vgg_2(target)
    loss_2 = torch.mean((extr_feat - real_feat) ** 2)
    extr_feat = vgg_5(image)
    real_feat = vgg_5(target)
    loss_5 = torch.mean((extr_feat - real_feat) ** 2)

    return ((2e-1)*loss_2 + (2e-2)*loss_5).cuda()


""" GAN generator and discriminator Losses """
def LossA(discriminator, device, output_g, target, optim_d, last_dx, last_dgz, lossT=False):
    if last_dx < 0.8 or last_dgz > 0.2:
        discriminator.train()
    else:
        discriminator.eval()
    batch_size = output_g.size(0)

    # Generator
    output_d = discriminator(output_g.detach()).view(-1).clamp(1e-7, 1-1e-7)
    d_g_z = output_d.mean().item()
    loss_g = -torch.log(output_d)
    if lossT:
        loss_g *= 2

    # Discriminator
    optim_d.zero_grad()
    output_t = discriminator(target.detach()).view(-1).clamp(1e-7, 1-1e-7)
    if output_t.cpu().detach().numpy().any() <= 0.0:
        print(output_t)
    d_x = output_t.mean().item()
    if last_dx < 0.8 or last_dgz > 0.2:
        loss_d = -torch.log(output_t) -torch.log(torch.full((batch_size,), 1., device=device) - output_d)
        if lossT:
            loss_d *= 2
        loss_d.mean().backward()

        optim_d.step()
    else:
        loss_d = torch.Tensor(np.zeros(1)).to(device)

    return loss_g.cuda(), loss_d.cuda(), d_x, d_g_z


""" Texture Loss """
def LossT(vgg, device, image, target, patch_size=16):
    # Images
    image = torch.split(image, 1, dim=0)
    target = torch.split(target, 1, dim=0)

    patches = []
    patches_target = []
    batch_size = int(image[0].shape[2] / patch_size) ** 2
    for el in image:
        new = el.unfold(1, 3, 3).unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        new = new.reshape((batch_size, 3, patch_size, patch_size))
        new = torch.split(new, 1, dim=0)
        patches += new
    del image
    for el in target:
        new = el.unfold(1, 3, 3).unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        new = new.reshape((batch_size, 3, patch_size, patch_size))
        new = torch.split(new, 1, dim=0)
        patches_target += new
    del target

    loss_1 = torch.Tensor(np.zeros(1)).cuda()
    loss_2 = torch.Tensor(np.zeros(1)).cuda()
    loss_3 = torch.Tensor(np.zeros(1)).cuda()

    for i, t in zip(patches, patches_target):
        loss_1 += torch.mean((gram_matrix(vgg[0](i)) - gram_matrix(vgg[0](t))) ** 2)
        loss_2 += torch.mean((gram_matrix(vgg[1](i)) - gram_matrix(vgg[1](t))) ** 2)
        loss_3 += torch.mean((gram_matrix(vgg[2](i)) - gram_matrix(vgg[2](t))) ** 2)

    loss_1 = loss_1.mean().item()
    loss_2 = loss_2.mean().item()
    loss_3 = loss_3.mean().item()

    return (3e-7 * loss_1 + 1e-6 * loss_2 + 1e-6 * loss_3).cuda()



