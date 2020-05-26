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

    return criterion(vgg(image), vgg(target.to(device))).cuda()


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



