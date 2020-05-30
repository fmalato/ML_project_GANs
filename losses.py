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
def LossA(discriminator, device, image, target):

    # Generator
    img = discriminator(image.to(device))
    loss_g = -torch.log(img).cuda()
    # Discriminator
    log1 = - torch.log(discriminator(target)).cuda()
    log2 = - torch.log(Tensor(np.ones(1)) - img).cuda()
    loss_d = log1 + log2
    if log1 > 0.25 or log2 > 0.25:
        train_d = True
    else:
        train_d = False
    # 2 if training PAT, 1 if training PA
    return 2 * loss_g.reshape(1), 2 * loss_d.reshape(1), train_d


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



