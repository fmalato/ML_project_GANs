import numpy as np
import os, re

from nets import FCNN
from PIL import Image
from math import floor
from torch.utils.data import DataLoader
from torchsummary import summary

from dataset import COCO

def PixelWiseMSELoss(input, target):
    # 1/whc * sum(w,h,c)(Iestwhc - Ihrwhc)**2
    out = (1 / input.shape[0] * input.shape[1] * input.shape[2]) * ((input - target)**2).sum(0).sum(1).sum(2)
    return out

def loadimg(fn, scale=4):
    try:
        img = Image.open(fn).convert('RGB')
    except IOError:
        return None
    w, h = img.size
    img.crop((0, 0, floor(w/scale), floor(h/scale)))
    img = img.resize((w//scale, h//scale), Image.ANTIALIAS)
    return np.array(img)/255

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


if __name__ == '__main__':
    """net = FCNN(input_channels=3)
    summary(net, input_size=(3, 32, 32))"""
    #data_train = COCO(root_dir='data/')

    imgs = os.listdir('train2014/')
    scale = 4

    for i in range(80000, 90000):
        orig = Image.open('train2014/{x}'.format(x=imgs[i]))
        if short_side(orig) >= 384:
            #print('Opening {x}'.format(x=imgs[i]))
            width, height = orig.size  # Get dimensions
            img = crop_central_square(orig)
            i_hr = img.resize((256, 256), Image.ANTIALIAS)
            i_lr = img.resize((int(256 / scale), int(256 / scale)))
            name = re.sub('\.jpg$', '', imgs[i])
            name = name.replace('COCO_train2014_', '')
            os.mkdir('data/{x}'.format(x=name))
            i_hr.save('data/{x}/hr.jpg'.format(x=name), 'JPEG')
            i_lr.save('data/{x}/lr.jpg'.format(x=name), 'JPEG')


