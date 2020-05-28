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
    valid = Variable(Tensor(np.ones(1)).long(), requires_grad=False)
    fake = Variable(Tensor(np.zeros(1)).long(), requires_grad=False)
    criterion = nn.CrossEntropyLoss().cuda()
    discriminator.cuda()
    # Generator
    img = discriminator(image.to(device))
    loss_g = criterion(img.to(device), valid.to(device))
    # Discriminator
    loss_d = criterion(discriminator(target), valid) + criterion(Tensor(np.ones(1)).long() - img, fake)

    return loss_g, loss_d


""" Texture Loss """
def LossT(vgg, device, image, target):
    criterion = nn.MSELoss()

    return criterion(gram_matrix(vgg(image)), gram_matrix(vgg(target.to(device)))).cuda()



