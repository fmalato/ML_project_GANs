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
from utils import crop_central_square, custom_bicubic, translate
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
    PER_CHANNEL_MEANS = np.zeros((3, 256, 256))
    PER_CHANNEL_MEANS[0].fill(0.47614917)
    PER_CHANNEL_MEANS[1].fill(0.45001204)
    PER_CHANNEL_MEANS[2].fill(0.40904046)
    img = Image.open(image_folder + 'lr/' + image_name)
    target = Image.open(image_folder + 'hr/' + image_name)
    tens = transforms.ToTensor()
    toimg = transforms.ToPILImage()

    target = tens(target)
    target = target.view((1, 3, 256, 256))
    #img.show()
    inp = tens(img)
    bicub_res = torch.from_numpy(np.asarray(img.resize((img.size[0] * 4, img.size[1] * 4), Image.BICUBIC)) / 255).view((1, 3, 256, 256))

    inp = inp.view((1, 3, 64, 64))
    output = net(inp, bicub_res)
    output = torch.add(output, torch.from_numpy(PER_CHANNEL_MEANS).view((1, 3, 256, 256))).clamp(0, 255)
    o = output.view((3, 256, 256))
    o = o.data.numpy()
    o = np.swapaxes(o, 0, 1)
    o = np.swapaxes(o, 1, 2)
    o = o * 255
    o = toimg(o.astype(np.uint8))
    bicub_res = np.asarray(img.resize((img.size[0] * 4, img.size[1] * 4), Image.BICUBIC)).astype(np.uint8)
    out = np.asarray(o).astype(np.uint8)
    c = 255 - out  # a temp uint8 array here
    np.putmask(bicub_res, c < bicub_res, c)  # a temp bool array here
    result = out + bicub_res
    result = Image.fromarray(result)
    result.show()

    # PSNR:
    score = psnr(output.detach().numpy() / 255, np.array(target) / 255)

    print('PSNR score for test image {x} is: %f'.format(x=image_name) % score)
    return score


if __name__ == '__main__':

    net = FCNN(input_channels=3)
    net.eval()
    tests = ['state_2e_E', 'state_3e_P']
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
    img = Image.open('evaluation/Set5/lr/baby.png')
    tens = transforms.ToTensor()
    pilimg = transforms.ToPILImage()
    img = custom_bicubic(tens(img).view((1, 3, img.size[0], img.size[1])), tens, pilimg, 4)
    img = img.view((3, 256, 256))
    img = pilimg(img)
    #lum = luminance(img)
    #img.show()
    #lum.show()


