import torch.nn as nn
from torchvision.models import vgg19


def LossE(device, image, target):
    criterion = nn.MSELoss()

    return criterion(image, target.to(device)).cuda()

def LossP(vgg, device, image, target):
    criterion = nn.MSELoss()

    return criterion(vgg(image), vgg(target.to(device))).cuda()

