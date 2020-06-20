import torch
import torchvision.transforms as transforms
import numpy as np
import os
import math
import matplotlib.pyplot as plt

from skimage import io
from skimage.transform import resize, downscale_local_mean, rescale

from FCNN_CPU import FCNN


def psnr(lr, hr):

    m, n, c = lr.shape[0], hr.shape[1], hr.shape[2]
    mse = np.sum((lr - hr) ** 2) / (m * n * c)

    return 10 * math.log10(1 / mse)

def test_single(net, img, target, image_name, model_name):
    net.eval()
    tens = transforms.ToTensor()
    inp = tens(img).float()

    inp = inp.view((1, 3, 56, 56))
    output = net(inp)

    o = output.view((3, 224, 224))
    o = o.data.numpy()
    o = np.swapaxes(o, 0, 1)
    o = np.swapaxes(o, 1, 2)

    bicub_res = rescale(img, (4, 4, 1), anti_aliasing=True)

    result = np.clip(o + bicub_res + 0.05, 0., 1.)
    # PSNR
    score = psnr(result, target)
    if image_name == "bird.png":
        fig, ax1 = plt.subplots(1, 1)
        ax1.imshow(result)
        ax1.set_title(model_name)
        plt.show()
        """fig, ax2 = plt.subplots(1, 1)
        ax2.imshow(img)
        ax2.set_title(model_name)
        plt.show()
        fig, ax3 = plt.subplots(1, 1)
        ax3.imshow(target)
        ax3.set_title(model_name)
        plt.show()"""

    print('PSNR score for test image {x} is: %f'.format(x=image_name) % score)
    return score


if __name__ == '__main__':

    net = FCNN(input_channels=3)
    net.eval()
    #tests = os.listdir('trained_models/')
    tests = ['state_1e_PA_Jun.pth', 'state_3e_PAT.pth', 'state_2e_E.pth', 'state_2e_P.pth']
    img_path = 'evaluation/Set5/'
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
            avg_psnr += test_single(net, img, target, image_name, el)
        avg_psnr = avg_psnr / len(img_dir)
        print('Average psnr score is: %f' % avg_psnr)
    img = io.imread('{p}{x}'.format(p=img_path, x='bird.png'))
    img = resize(img, (56, 56), anti_aliasing=True)
    # TODO: check rescaling
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
