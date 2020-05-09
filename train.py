import numpy as np
import os, re
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from nets import FCNN
from PIL import Image
from math import floor
from torch.utils.data import DataLoader
from torchsummary import summary
from torch import FloatTensor

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

def generate_dataset(scale=4):
    imgs = os.listdir('train2014/')

    for i in range(90000):
        orig = Image.open('train2014/{x}'.format(x=imgs[i]))
        if short_side(orig) >= 384:
            # print('Opening {x}'.format(x=imgs[i]))
            width, height = orig.size  # Get dimensions
            img = crop_central_square(orig)
            i_hr = img.resize((256, 256), Image.ANTIALIAS)
            i_lr = img.resize((int(256 / scale), int(256 / scale)))
            name = re.sub('\.jpg$', '', imgs[i])
            name = name.replace('COCO_train2014_', '')
            os.mkdir('data/{x}'.format(x=name))
            i_hr.save('data/{x}/hr.jpg'.format(x=name), 'JPEG')
            i_lr.save('data/{x}/lr.jpg'.format(x=name), 'JPEG')

def square_patch(img_path, size=32):
    img = Image.open(img_path)
    patches = []
    scale = int(img.size[0] / size)
    # Quadratic time, but since it will be used to scale 64x64 images to 32x32 patches, it's viable.
    for i in range(scale):
        for j in range(scale):
            patches.append(img.crop((i * size, j * size, (i + 1) * size, (j + 1) * size)))
    return patches

def train(net, criterion, optimizer, epoch, batch_size=16, steps=50):
    net.train()
    # TODO: how to use this on the GPU via ssh?
    images = os.listdir('data/')
    for e in range(epoch):
        print('Epoch %d' % (e))
        for h in range(steps):
            avg_loss = 0
            data = []
            gt = []
            for n in range(batch_size):
                idx = random.randint(0, len(images))
                res_d = square_patch('data/{x}/lr.jpg'.format(x=images[idx]), 32)
                res_g = square_patch('data/{x}/hr.jpg'.format(x=images[idx]), 128)
                print(idx)
                for x in res_d:
                    data.append(x)
                for x in res_g:
                    gt.append(x)

            for i in range(len(data)):
                # TODO: random dimensional error sometimes, maybe a bad image?
                optimizer.zero_grad()
                input = FloatTensor(np.array(data[i]))
                input = input.view((-1, 3, 32, 32))
                output = net(input)
                target = FloatTensor(np.array(gt[i]))
                target = target.view((-1, 3, 128, 128))
                loss = (1 / (3 * 32 * 32)) * criterion(output, target)
                avg_loss += loss.detach().item()
                loss.backward()
                optimizer.step()

            print('Epoch: %d - Step: %d, Avg. Loss: %f' % (e, h, avg_loss / (batch_size * 4)))

        print('Saving checkpoint.')
        torch.save(net.state_dict(), 'state_{d}e_50s.pth'.format(d=e))
    trans = transforms.ToPILImage()
    out = trans(output.view((3, 128, 128)))
    out.show()
    target = trans(target.view((3, 128, 128)))
    target.show()


if __name__ == '__main__':
    net = FCNN(input_channels=3)
    summary(net, input_size=(3, 32, 32))

    train(net, nn.MSELoss(), optim.Adam(net.parameters(), lr=1e-4), epoch=20, batch_size=4)






