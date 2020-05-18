import torch.nn as nn
from torchvision.models import vgg19


def LossE(image, target):
    criterion = nn.MSELoss()

    return criterion(image, target)

def LossP(image, target):
    criterion = nn.MSELoss()
    net = vgg19(pretrained=True, progress=False)

    return criterion(net(image), net(target))

