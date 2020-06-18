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

def load_data(data_folder, batch_size, train, kwargs):
    transform = {
        'train': transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.47614917, 0.45001204, 0.40904046],
                                     std=[0.229, 0.224, 0.225])
            ]),
        'test': transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.47614917, 0.45001204, 0.40904046],
                                     std=[0.229, 0.224, 0.225])
            ])
        }
    data = datasets.ImageFolder(root=data_folder, transform=transform['train' if train else 'test'])
    data_loader = torch.utils.data.DataLoader(data,
                                              batch_size=batch_size,
                                              shuffle=True, **kwargs,
                                              drop_last=True if train else False)
    return data_loader

def loadimg(fn, scale=4):
    try:
        img = Image.open(fn).convert('RGB')
    except IOError:
        return None
    w, h = img.size
    img.crop((0, 0, floor(w/scale), floor(h/scale)))
    img = img.resize((w//scale, h//scale), Image.ANTIALIAS)
    return np.array(img)/255


def square_patch(img_path, size=32):
    img = Image.open(img_path)
    patches = []
    scale = int(img.size[0] / size)
    # Quadratic time, but since it will be used to scale 64x64 images to 32x32 patches, it's viable.
    for i in range(scale):
        for j in range(scale):
            patches.append(img.crop((i * size, j * size, (i + 1) * size, (j + 1) * size)))
    return patches


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

    G = torch.mm(features, features.t())  # compute the gram product

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

def img_to_pt():
    if not os.path.exists('data_pt/'):
        os.mkdir('data_pt/')
        os.mkdir('data_pt/train/')
        os.mkdir('data_pt/target/')
        os.mkdir('data_pt/bicub/')
    imgs = os.listdir('data/train')
    i = 0
    for el in imgs:
        if i % 100 == 0:
            print('Processed: %d/%d' % (i, len(imgs)))
        name = os.path.splitext(el)[0]
        image = io.imread('data/train/{x}'.format(x=el))
        target = io.imread('data/target/{x}'.format(x=el))
        bicub = resize(image, (128, 128), anti_aliasing=True)

        image = np.array(image, dtype=np.float64) / 255
        target = np.array(target, dtype=np.float64) / 255

        image = np.swapaxes(image, 2, 1)
        target = np.swapaxes(target, 2, 1)
        bicub = np.swapaxes(bicub, 2, 1)
        image = np.swapaxes(image, 1, 0)
        target = np.swapaxes(target, 1, 0)
        bicub = np.swapaxes(bicub, 1, 0)

        image = torch.from_numpy(image)
        target = torch.from_numpy(target)
        bicub = torch.from_numpy(bicub)

        torch.save(image, 'data_pt/train/{n}.pt'.format(n=name))
        torch.save(target, 'data_pt/target/{n}.pt'.format(n=name))
        torch.save(bicub, 'data_pt/bicub/{n}.pt'.format(n=name))
        i += 1

#img_to_pt()