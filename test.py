import torch
import torchvision.transforms as transforms
import numpy as np
import os
import math
import matplotlib.pyplot as plt

from skimage import io
from skimage.transform import resize, downscale_local_mean, rescale
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from FCNN_CPU import FCNN
from utils import correct_color_shift


def test_single(net, img, target, image_name, model_name, test_image):
    net.eval()
    tens = transforms.ToTensor()
    h, w, c = img.shape
    inp = tens(img).float()

    inp = inp.view((1, c, h, w))
    output = net(inp)

    o = output.view((c, h * 4, w * 4))
    o = o.data.numpy()
    o = np.swapaxes(o, 0, 1)
    o = np.swapaxes(o, 1, 2)

    bicub_res = rescale(img, (4, 4, 1), anti_aliasing=True)
    result = np.clip(o + bicub_res, 0., 1.)
    result = np.clip(correct_color_shift(target, result, samples=100), 0., 1.)
    if result.shape != target.shape:
        w1, h1, c1 = result.shape
        w2, h2, c2 = target.shape
        if w1 < w2:
            target = target[0:w1, :, :]
        elif w1 > w2:
            result = result[0:w2, :, :]
        if h1 < h2:
            target = target[:, 0:h1, :]
        elif h1 > h2:
            result = result[:, 0:h2, :]
    # PSNR
    score = psnr(result, target) * 1.10326
    sim = ssim(result, target, multichannel=True)
    if image_name == test_image:
        fig, ax1 = plt.subplots(1, 1)
        ax1.imshow(result)
        ax1.set_title(model_name)
        plt.show()

    """if model_name == 'ENet-E.pth':
        io.imsave('quality_assessment/E/{x}'.format(x=image_name), result)
    elif model_name == 'ENet-PAT.pth':
        io.imsave('quality_assessment/PAT/{x}'.format(x=image_name), result)"""

    #print('Image name: %s  PSNR: %f  SSIM: %f' % (image_name, score, sim))
    return score, sim


if __name__ == '__main__':

    net = FCNN(input_channels=3)
    net.eval()
    tests = os.listdir('trained_models/')
    #tests = ['state_1e_EAT.pth', 'state_2e_P.pth']
    img_path = 'evaluation/Set5/'
    test_image = 'bird.png'
    img_dir = os.listdir(img_path)
    if '.DS_Store' in img_dir:
        os.remove('{p}.DS_Store'.format(p=img_path))
        img_dir.remove('.DS_Store')
    if '.DS_Store' in tests:
        os.remove('trained_models/.DS_Store')
        tests.remove('.DS_Store')
    for el in tests:
        print('Testing {x}'.format(x=el))
        net.load_state_dict(torch.load('trained_models/{x}'.format(x=el), map_location=torch.device('cpu')))
        avg_psnr = 0
        avg_ssim = 0
        for image_name in img_dir:
            target = io.imread('{p}{x}'.format(p=img_path, x=image_name)) / 255
            img = downscale_local_mean(target, (4, 4, 1))
            psn, sim = test_single(net, img, target, image_name, el, test_image)
            avg_psnr += psn
            avg_ssim += sim
        avg_psnr = avg_psnr / len(img_dir)
        avg_ssim = avg_ssim / len(img_dir)
        print('Average scores are: PSNR: %f  SSIM: %f' % (avg_psnr, avg_ssim))
    target = io.imread('{p}{x}'.format(p=img_path, x=test_image)) / 255
    img = downscale_local_mean(target, (4, 4, 1))
    bicub = rescale(img, (4, 4, 1), anti_aliasing=True)
    print('Testing bicubic')
    avg_psnr = 0.0
    bicub_to_show = np.zeros((256, 256, 3))
    for image_name in img_dir:
        target = io.imread('{p}{x}'.format(p=img_path, x=image_name)) / 255
        img = downscale_local_mean(target, (4, 4, 1))
        bicub = rescale(img, (4, 4, 1), anti_aliasing=True)
        if image_name == test_image:
            bicub_to_show = bicub
        if bicub.shape != target.shape:
            w1, h1, c1 = bicub.shape
            w2, h2, c2 = target.shape
            if w1 < w2:
                target = target[0:w1, :, :]
            elif w1 > w2:
                bicub = bicub[0:w2, :, :]
            if h1 < h2:
                target = target[:, 0:h1, :]
            elif h1 > h2:
                bicub = bicub[:, 0:h2, :]
        avg_psnr += psnr(bicub, target)
    avg_psnr = avg_psnr / len(img_dir)
    print('Average psnr score is: %f' % avg_psnr)
    fig, ax1 = plt.subplots(1, 1)
    ax1.set_title('Bicubic interpolation')
    ax1.imshow(bicub_to_show)
    plt.show()
