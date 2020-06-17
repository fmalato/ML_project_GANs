import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math
import matplotlib.pyplot as plt

from torch import FloatTensor
from PIL import Image
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean

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

def psnr(lr, hr):

    m, n, c = lr.shape[0], hr.shape[1], hr.shape[2]
    mse = np.sum((lr - hr) ** 2) / (m * n * c)

    return 10 * math.log10(1 / mse)

def test_single(net, img, target, image_name):
    net.eval()
    tens = transforms.ToTensor()
    toimg = transforms.ToPILImage()

    target = np.array(target)
    #img.show()
    inp = tens(img).float()

    inp = inp.view((1, 3, 56, 56))
    output = net(inp).clamp(0, 255)
    #output = torch.add(output, torch.from_numpy(PER_CHANNEL_MEANS).view((1, 3, 256, 256))).clamp(0, 255)
    o = output.view((3, 224, 224))
    o = o.data.numpy()
    o = np.swapaxes(o, 0, 1)
    o = np.swapaxes(o, 1, 2)
    """o = o * 255
    o = toimg(o.astype(np.uint8))"""
    bicub_res = resize(img, (224, 224), anti_aliasing=True)
    """bicub_res = (bicub_res * 255).astype(np.uint8)
    out = np.asarray(o).astype(np.uint8)
    c = 255 - out  # a temp uint8 array here
    np.putmask(bicub_res, c < bicub_res, c)  # a temp bool array here"""
    result = np.clip(o + bicub_res, 0., 1.)
    # PSNR:
    score = psnr(result, np.array(target))
    #result = Image.fromarray(result)
    if image_name == "zebra.png":
        io.imshow(result)
        plt.show()

    """if image_name == 'bird.png':
        output_img = toimg(output.detach().numpy().reshape((256, 256, 3)).astype(np.uint8) * 255)
        output_img.show()"""
    print('PSNR score for test image {x} is: %f'.format(x=image_name) % score)
    return score


if __name__ == '__main__':

    net = FCNN(input_channels=3)
    net.eval()
    #tests = os.listdir('trained_models/')
    tests = ['state_3e_E_2.pth', 'state_3e_P_Jun.pth', 'state_1e_PA_Jun.pth']
    img_path = 'evaluation/Set14/'
    if '.DS_Store' in tests:
        os.remove('trained_models/.DS_Store')
        tests.remove('.DS_Store')
    for el in tests:
        print('Testing {x}'.format(x=el))
        net.load_state_dict(torch.load('trained_models/{x}'.format(x=el), map_location=torch.device('cpu')))
        avg_psnr = 0
        img_dir = os.listdir(img_path)
        if '.DS_Store' in img_dir:
            os.remove('{p}.DS_Store'.format(p=img_path))
            img_dir.remove('.DS_Store')
        for image_name in img_dir:
            img = io.imread('{p}{x}'.format(p=img_path, x=image_name))
            img = resize(img, (224, 224), anti_aliasing=True)
            target = img
            img = downscale_local_mean(img, (4, 4, 1))
            avg_psnr += test_single(net, img, target, image_name)
        avg_psnr = avg_psnr / len(img_dir)
        print('Average psnr score is: %f' % avg_psnr)
