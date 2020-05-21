import torch.nn as nn
from torchvision.models import vgg19


def LossE(image, target):
    criterion = nn.MSELoss()

    return criterion(image, target).cuda()

def LossP(vgg, image, target):
    criterion = nn.MSELoss()

    return criterion(vgg(image), vgg(target)).cuda()

