import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math

from torch import FloatTensor
from PIL import Image

from FCNN_CPU import FCNN
from utils import crop_central_square, custom_bicubic
from losses import LossE, LossP, LossA, LossT

def test_single(net, image_folder, image_name, criterion):
    net.eval()
    img = Image.open(image_folder + 'lr/' + image_name)
    target = Image.open(image_folder + 'hr/' + image_name)
    tens = transforms.ToTensor()

    target = tens(target)
    target = target.view((1, 3, 256, 256))
    input = tens(img)

    input = input.view((1, 3, 64, 64))
    output = net(input)

    loss = criterion(target, output)
    # PSNR from review
    psnr = 10 * math.log10((255**2) / loss.item())

    # PSNR from tensorflow source code
    #psnr = 20 * math.log(255) / math.log(10.0) - np.float32(10 / np.log(10)) * math.log(loss)

    trans = transforms.ToPILImage(mode='RGB')
    if image_name == 'bird.png':
        output = output.view((3, 256, 256))
        output = trans(output)
        output.show(title="Guessing")
    print('PSNR score for test image {x} is: %f'.format(x=image_name) % psnr)
    return psnr


if __name__ == '__main__':

    net = FCNN(input_channels=3)
    net.eval()
    tests = ['state_2e_LossE', 'state_1e_LossP', 'state_1e_EA', 'state_1e_PA', 'state_1e_PAT']
    for el in tests:
        print('Testing {x}'.format(x=el))
        net.load_state_dict(torch.load('trained_models/{x}.pth'.format(x=el), map_location=torch.device('cpu')))
        avg_psnr = 0
        for image_name in os.listdir('evaluation/Set5/lr'):
            img = Image.open('evaluation/Set5/lr/{x}'.format(x=image_name))
            target = Image.open('evaluation/Set5/hr/{x}'.format(x=image_name))
            avg_psnr += test_single(net, 'evaluation/Set5/', image_name, criterion=nn.MSELoss())
        avg_psnr = avg_psnr / len(os.listdir('evaluation/Set5/lr'))
        print('Average psnr score is: %f' % avg_psnr)
    img = Image.open('evaluation/Set5/lr/bird.png')
    tens = transforms.ToTensor()
    pilimg = transforms.ToPILImage()
    img = custom_bicubic(tens(img).view((1, 3, img.size[0], img.size[1])), tens, pilimg, 4)
    img = img.view((3, 256, 256))
    img = pilimg(img)
    img.show()


