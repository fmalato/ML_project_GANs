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


def test_single(net, img, target, image_name, model_name):
    net.eval()
    tens = transforms.ToTensor()
    PER_CHANNEL_MEANS = np.array([0.47614917, 0.45001204, 0.40904046])
    inp = tens(img).float()
    if model_name == 'state_4e_PA.pth':
        inp = tens(img - PER_CHANNEL_MEANS).float()

    inp = inp.view((1, 3, 56, 56))
    output = net(inp)

    o = output.view((3, 224, 224))
    o = o.data.numpy()
    o = np.swapaxes(o, 0, 1)
    o = np.swapaxes(o, 1, 2)

    bicub_res = rescale(img, (4, 4, 1), anti_aliasing=True)

    result = np.clip(o + bicub_res, 0., 1.)
    # PSNR
    score = psnr(result, target)
    sim = ssim(result, target, multichannel=True)
    if image_name == "bird.png":
        fig, ax1 = plt.subplots(1, 1)
        ax1.imshow(result)
        ax1.set_title(model_name)
        plt.show()

    print('Image name: %s  PSNR: %f  SSIM: %f' % (image_name, score, sim))
    return score, sim


if __name__ == '__main__':

    net = FCNN(input_channels=3)
    net.eval()
    #tests = os.listdir('trained_models/')
    tests = ['ENet-PA.pth', 'ENet-PAT.pth', 'state_4e_PA.pth', 'ENet-E.pth', 'state_3e_EA.pth']
    img_path = 'evaluation/Set5/'
    if '.DS_Store' in tests:
        os.remove('trained_models/.DS_Store')
        tests.remove('.DS_Store')
    for el in tests:
        print('Testing {x}'.format(x=el))
        net.load_state_dict(torch.load('trained_models/{x}'.format(x=el), map_location=torch.device('cpu')))
        avg_psnr = 0
        avg_ssim = 0
        img_dir = os.listdir(img_path)
        if '.DS_Store' in img_dir:
            os.remove('{p}.DS_Store'.format(p=img_path))
            img_dir.remove('.DS_Store')
        for image_name in img_dir:
            img = io.imread('{p}{x}'.format(p=img_path, x=image_name))
            img = resize(img, (224, 224), anti_aliasing=True)
            target = img
            img = downscale_local_mean(img, (4, 4, 1))
            psn, sim = test_single(net, img, target, image_name, el)
            avg_psnr += psn
            avg_ssim += sim
        avg_psnr = avg_psnr / len(img_dir)
        avg_ssim = avg_ssim / len(img_dir)
        print('Average scores are: PSNR: %f  SSIM: %f' % (avg_psnr, avg_ssim))
    img = io.imread('{p}{x}'.format(p=img_path, x='bird.png'))
    img = resize(img, (56, 56), anti_aliasing=True)
    print('Testing bicubic')
    bicub_res = rescale(img, (4, 4, 1), anti_aliasing=True)
    avg_psnr = 0.0
    for image_name in img_dir:
        img = io.imread('{p}{x}'.format(p=img_path, x=image_name))
        img = resize(img, (224, 224), anti_aliasing=True)
        target = img
        img = downscale_local_mean(img, (4, 4, 1))
        bicub = rescale(img, (4, 4, 1), anti_aliasing=True)
        avg_psnr += psnr(bicub, target)
    avg_psnr = avg_psnr / len(img_dir)
    print('Average psnr score is: %f' % avg_psnr)
    fig, ax1 = plt.subplots(1, 1)
    ax1.imshow(bicub_res)
    plt.show()
