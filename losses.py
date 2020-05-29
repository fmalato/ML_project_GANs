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

    return 0.2*loss_2 + loss_5


""" GAN generator and discriminator Losses """
def LossA(discriminator, device, image, target):
    valid = Variable(Tensor(np.ones(1)).long(), requires_grad=False).cuda()
    fake = Variable(Tensor(np.zeros(1)).long(), requires_grad=False).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    discriminator.cuda()
    # Generator
    img = discriminator(image.to(device))
    loss_g = criterion(img.to(device), valid.to(device))
    # Discriminator
    loss_d = criterion(discriminator(target), valid) + criterion(Tensor(np.ones(1)).long().cuda() - img, fake)

    return loss_g, loss_d


""" Texture Loss """
def LossT(vgg, device, image, target):
    criterion = nn.MSELoss()
    criterion.cuda()
    loss = 0.0
    vgg_2 = vgg[0]
    vgg_5 = vgg[1]
    vgg_5.cuda()
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
    for i in range(batch_size):
        loss += criterion(gram_matrix(vgg_5(pat_list[i].view((1, 3, 16, 16)))).to(device),
                          gram_matrix(vgg_5(pat_tar_list[i].view((1, 3, 16, 16)))).to(device))
    loss = torch.div(loss, batch_size)

    return loss



