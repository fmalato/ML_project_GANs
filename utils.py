import random
import os
import torch

import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

from torchvision import datasets
from math import floor
from PIL import Image, PngImagePlugin
from skimage import io
from skimage.transform import resize

# Given that I have to perform the same random crop on both the image and the target, I had to redefine the function
def random_crop(image, target, image_max_range=32, target_scale=4):
    crop_x = random.randint(0, image_max_range)
    crop_y = random.randint(0, image_max_range)
    image = image.crop((crop_x,
                       crop_y,
                       crop_x + image_max_range,
                       crop_y + image_max_range
                        ))
    target = target.crop((crop_x * target_scale,
                         crop_y * target_scale,
                         (crop_x + image_max_range) * target_scale,
                         (crop_y + image_max_range) * target_scale
                          ))
    return image, target

def generate_dataset(src, dst, scale=4):
    imgs = os.listdir(src)
    if not os.path.exists(dst + '/target/'):
        os.mkdir(dst + '/target/')
    if not os.path.exists(dst + '/train/'):
        os.mkdir(dst + '/train/')

    for i in range(len(imgs)):
        name = os.path.splitext(imgs[i])[0]
        orig = Image.open(src + '/{x}'.format(x=imgs[i]))
        img = crop_central_square(orig)
        i_hr = img.resize((256, 256), Image.ANTIALIAS)
        i_lr = img.resize((int(256 / scale), int(256 / scale)), Image.ANTIALIAS)
        patches_hr = img_square_patch(i_hr, size=128)
        patches_lr = img_square_patch(i_lr, size=32)
        for j in range(len(patches_hr)):
            patches_hr[j].save(dst + '/target/{x}_{idx}.jpg'.format(x=name, idx=j))
        for j in range(len(patches_lr)):
            patches_lr[j].save(dst + '/train/{x}_{idx}.jpg'.format(x=name, idx=j))
        if i % 100 == 0:
            print('Processed: %d / %d' % (i, len(imgs)))

    print('Done.')

def short_side(img):
    w, h = img.size
    if w < h:
        return w
    else:
        return h

def crop_central_square(img):
    w, h = img.size
    if w < h:
        return img.crop((0, floor((h - w) / 2), w, floor(((h - w) / 2) + w)))
    else:
        return img.crop((floor((w - h) / 2), 0, floor(((w - h) / 2) + h), h))

def crop_central_square_np(img):
    w, h, c = img.shape
    if w < h:
        return img[0:w, floor((h - w) / 2):floor(((h - w) / 2) + w), :]
    else:
        return img[floor((w - h) / 2):floor(((w - h) / 2) + h), 0:h, :]


def img_square_patch(img, size=32):
    patches = []
    scale = int(img.size[0] / size)
    # Quadratic time, but since it will be used to scale 64x64 images to 32x32 patches, it's viable.
    for i in range(scale):
        for j in range(scale):
            patches.append(img.crop((i * size, j * size, (i + 1) * size, (j + 1) * size)))
    return patches


def init_weights(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(model.weight.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

def remove_grayscale():
    PngImagePlugin.MAX_TEXT_CHUNK = 1000 * (1024**2)
    for el in os.listdir('data/train/'):
        img = Image.open('data/train/{x}'.format(x=el))
        if len(img.getbands()) != 3:
            os.remove('data/train/{x}'.format(x=el))
            os.remove('data/target/{x}'.format(x=el))
            print('{x} removed.'.format(x=el))

def gram_matrix(input):
    """ From PyTorch tutorials """
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = features.mm(features.t())#.div(a * b * c * d)  # compute the gram product

    return G

def custom_bicubic(input_tensor, transf_to_tensor, transf_to_img, scale_factor=4):

    batch_size, channels, w, h = input_tensor.shape
    x = input_tensor.view((channels, w, h))
    x_img = transf_to_img(x)
    x_img = x_img.resize((w * scale_factor, h * scale_factor), Image.BICUBIC)
    x_t = transf_to_tensor(x_img)
    x_t = x_t.view((batch_size, channels, w * scale_factor, h * scale_factor))

    return x_t


def translate(x, mx):
    lo = x.min()
    rng = x.max()-lo
    return (x-lo)*mx/rng


def generate_data():
    print('Generating from train2014')
    generate_dataset('../coco/train2014/', 'data/')
    print('Generating from val2014')
    generate_dataset('../coco/val2014/', 'data/')
    print('Generating from test2014')
    generate_dataset('../coco/test2014/', 'data/')
    print('Removing grayscale images')
    remove_grayscale()


def img_to_pt(chunk_size=16):
    PngImagePlugin.MAX_TEXT_CHUNK = 1000 * (1024 ** 2)
    if not os.path.exists('data_pt/'):
        os.mkdir('data_pt/')
        os.mkdir('data_pt/train/')
        os.mkdir('data_pt/target/')
        os.mkdir('data_pt/bicub/')
    imgs = os.listdir('data/train')
    i = 1
    images = []
    targets = []
    bicubs = []
    for el in imgs:
        if i % 1000 == 0:
            print('Processed: %d/%d' % (i, len(imgs)))
        image = io.imread('data/train/{x}'.format(x=el)) / 255
        target = io.imread('data/target/{x}'.format(x=el)) / 255
        bicub = resize(image, (128, 128))

        image = np.swapaxes(image, 2, 1)
        target = np.swapaxes(target, 2, 1)
        bicub = np.swapaxes(bicub, 2, 1)
        image = np.swapaxes(image, 1, 0)
        target = np.swapaxes(target, 1, 0)
        bicub = np.swapaxes(bicub, 1, 0)

        images.append(torch.from_numpy(image).view((1, 3, 32, 32)))
        targets.append(torch.from_numpy(target).view((1, 3, 128, 128)))
        bicubs.append(torch.from_numpy(bicub).view((1, 3, 128, 128)))
        if i % chunk_size == 0 and i is not 0:
            tens_i = torch.cat(images)
            tens_t = torch.cat(targets)
            tens_b = torch.cat(bicubs)
            name = 'chunk_{i}'.format(i=i)
            torch.save(tens_i, 'data_pt/train/{n}.pt'.format(n=name))
            torch.save(tens_t, 'data_pt/target/{n}.pt'.format(n=name))
            torch.save(tens_b, 'data_pt/bicub/{n}.pt'.format(n=name))
            images = []
            targets = []
            bicubs = []

        i += 1


def print_stats(is_adv, current_epoch, epochs, num_batches, current_step, losses, losses_g, losses_d, D_xs, D_gs, step_update):
    if is_adv:
        print('Epoch %d/%d - Step: %d/%d  Loss G: %f (%f) Loss D: %f  D(x): %f  D(G(z)): %f' % (current_epoch+1, epochs,
                                                                                                current_step, num_batches,
                                                                                                sum(losses) / step_update,
                                                                                                sum(losses_g) / step_update,
                                                                                                sum(losses_d) / step_update,
                                                                                                sum(D_xs) / step_update,
                                                                                                sum(D_gs) / step_update))
    else:
        print('Epoch %d/%d - Step: %d/%d  Loss G: %f' % (current_epoch+1, epochs,
                                                         current_step, num_batches,
                                                         sum(losses) / step_update))


def time_stats(epoch_times, end_step, start_step, num_imgs, i):
    hours, rem = divmod((sum(epoch_times) / len(epoch_times)) * (num_imgs - i) / 100, 3600)
    minutes, seconds = divmod(rem, 60)
    print('Time for the last step: {:05.2f} s    Epoch ETA: {:0>2}:{:0>2}:{:0>2}'.format(
        end_step - start_step,
        int(hours),
        int(minutes),
        int(seconds)))


def true_or_false(labels):
    result = []
    for el in labels:
        if el < 0.5:
            result.append(0)
        else:
            result.append(1)
    return result


def correct_color_shift(reference, shifted, samples=50):
    wr, hr, cr = reference.shape
    randx = []
    randy = []
    # take 15 random pixels
    for i in range(samples):
        randx.append(random.randint(0, wr - 1))
        randy.append(random.randint(0, hr - 1))
    mean_p = np.zeros(cr)
    for (x, y) in zip(randx, randy):
        mean_p += reference[x, y, :] - shifted[x, y, :]
    mean_p /= samples
    return shifted + mean_p


def compute_patches(image, target, patch_size=16, step=8):
    # Images
    image = torch.split(image, 1, dim=0)
    target = torch.split(target, 1, dim=0)

    patches = []
    patches_target = []
    batch_size = int(image[0].shape[2] / patch_size) ** 2
    for el in image:
        new = el.unfold(1, 3, 3).unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        new = new.reshape((batch_size, 3, patch_size, patch_size))
        new = torch.split(new, 1, dim=0)
        for idx in range(len(new)):
            if idx % step == 0:
                patches.append(new[idx])
    del image
    for el in target:
        new = el.unfold(1, 3, 3).unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        new = new.reshape((batch_size, 3, patch_size, patch_size))
        new = torch.split(new, 1, dim=0)
        for idx in range(len(new)):
            if idx % step == 0:
                patches.append(new[idx])
    del target

    return patches, patches_target


import matplotlib.pyplot as plt

folders = ['bicubic', 'ENet-E', 'ENet-P', 'ENet-EA', 'ENet-PA', 'ENet-EAT', 'ENet-PAT']
images = []
for folder in folders:
    images.append(plt.imread('results_reconstr/{x}/bird.png'.format(x=folder))[2:146, 118:262, :])


