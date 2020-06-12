import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math

from torch import FloatTensor
from PIL import Image
from skimage import io

from FCNN_CPU import FCNN
from utils import crop_central_square, custom_bicubic
from losses import LossE, LossP, LossA, LossT


def luminance(img):
    lum_img = np.zeros((img.size[0], img.size[1]), dtype=np.uint8)
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            pixelRGB = img.getpixel((i, j))
            R, G, B = pixelRGB
            lum_img[j][i] = math.sqrt(0.47614917*(R**2) + 0.45001204*(G**2) + 0.40904046*(B**2))

    return Image.fromarray((lum_img * 255).astype(np.uint8), 'L')

def psnr(lr_path, hr_path):
    #lr = io.imread(lr_path).astype(np.float32) / 255
    #hr = io.imread(hr_path).astype(np.float32) / 255

    lr = lr_path
    hr = hr_path

    m, n, c = lr.shape[0], hr.shape[1], hr.shape[2]
    mse = np.sum((lr - hr) ** 2) / (m * n * c)

    return 10 * math.log10(1 / mse)

def test_single(net, image_folder, image_name, criterion):
    net.eval()
    PER_CHANNEL_MEANS = np.array([0.47614917, 0.45001204, 0.40904046])
    img = Image.open(image_folder + 'lr/' + image_name)
    target = Image.open(image_folder + 'hr/' + image_name)
    tens = transforms.ToTensor()
    toimg = transforms.ToPILImage()

    target = tens(target)
    target = target.view((1, 3, 256, 256))
    input = tens(img)
    bicub_res = tens(img.resize((img.size[0] * 4, img.size[1] * 4), Image.ANTIALIAS))

    input = input.view((1, 3, 64, 64))
    output = net(input, bicub_res)
    output = torch.add(output, bicub_res).clamp(0, 255)


    #loss = criterion(tens(res).view((1, 3, 256, 256)), output)

    # PSNR:
    score = psnr(output.detach().numpy() / 255, np.array(target) / 255)
    # PSNR from review
    #psnr = 10 * math.log10((255**2) / loss.item())

    # PSNR from tensorflow source code
    #psnr = 20 * math.log(255) / math.log(10.0) - np.float32(10 / np.log(10)) * math.log(loss)

    trans = transforms.ToPILImage(mode='RGB')
    if image_name == 'bird.png':
        # TODO: get the right transform
        output_img = (np.asarray(toimg(output.view(3, 256, 256))) + PER_CHANNEL_MEANS)
        output_img = Image.fromarray(output_img.astype(np.uint8))
        output = output.view((3, 256, 256))
        output = trans(output)
        output_img.show()
        output.show(title="Guessing")
    print('PSNR score for test image {x} is: %f'.format(x=image_name) % score)
    return score


if __name__ == '__main__':

    net = FCNN(input_channels=3)
    net.eval()
    tests = ['state_1e_E']
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
    #lum = luminance(img)
    img.show()
    #lum.show()


