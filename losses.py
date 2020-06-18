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
    criterion = nn.MSELoss().cuda()
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
    criterion = nn.MSELoss()
    vgg_1 = vgg[0].cuda()
    vgg_2 = vgg[1].cuda()
    vgg_3 = vgg[2].cuda()
    # Images
    image = torch.split(image, 1, dim=0)
    img_size = image[0].shape[2]
    batch_size = int(img_size / patch_size)**2
    patches = []
    for el in image:
        new = el.unfold(1, 3, 3).unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        new = new.reshape((batch_size, 3, patch_size, patch_size))
        patches.append(new)
    patches = torch.cat(patches).to(device)
    # Targets
    img_size = target[0].shape[2]
    batch_size = int(img_size / patch_size) ** 2
    target = torch.split(target, 1, dim=0)
    patches_target = []
    for el in target:
        new = el.unfold(1, 3, 3).unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        new = new.reshape((batch_size, 3, patch_size, patch_size))
        patches_target.append(new)
    patches_target = torch.cat(patches_target).to(device)
    # Computing loss
    loss_1 = criterion(gram_matrix(vgg_1(patches)).float(), gram_matrix(vgg_1(patches_target.float())).float())
    loss_2 = criterion(gram_matrix(vgg_2(patches)).float(), gram_matrix(vgg_2(patches_target.float())).float())
    loss_3 = criterion(gram_matrix(vgg_3(patches)).float(), gram_matrix(vgg_3(patches_target.float())).float())

    return (3e-7 * loss_1 + 1e-6 * loss_2 + 1e-6 * loss_3).cuda()



