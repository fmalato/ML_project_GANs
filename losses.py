import torch.nn as nn
import math

from utils import gram_matrix
from torch import FloatTensor


""" Baseline MSE Loss """
def LossE(device, image, target):
    criterion = nn.MSELoss()

    return criterion(image, target.to(device)).cuda()


""" Perceptual Loss """
def LossP(vgg, device, image, target):
    criterion = nn.MSELoss()
    #criterion.cuda()
    vgg_2 = vgg[0]
    vgg_5 = vgg[1]
    #vgg_2.cuda()
    #vgg_5.cuda()

    extr_feat = vgg_2(image)
    real_feat = vgg_2(target)
    loss_2 = criterion(extr_feat, real_feat.detach())
    extr_feat = vgg_5(image)
    real_feat = vgg_5(target)
    loss_5 = criterion(extr_feat, real_feat.detach())

    return 0.2*loss_2 + loss_5


""" GAN generator and discriminator Losses """
def LossA(discriminator, image, target):

    criterion = nn.CrossEntropyLoss()
    loss_g = criterion(discriminator(image), FloatTensor(0))
    loss_d = -(math.log(discriminator(target))) - (math.log(1 - discriminator(image)))

    return loss_g, loss_d


""" Texture Loss """
def LossT(vgg, device, image, target):
    criterion = nn.MSELoss()

    return criterion(gram_matrix(vgg(image)), gram_matrix(vgg(target.to(device)))).cuda()



