import numpy as np
import os, re
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from nets import FCNN
from PIL import Image
from math import floor
from torch.utils.data import DataLoader
from torchsummary import summary
from torch import FloatTensor
from torchvision import datasets
from torchvision.datasets import ImageFolder

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
    os.mkdir('data/train')
    os.mkdir('data/target')

    for i in range(90000):
        orig = Image.open('train2014/{x}'.format(x=imgs[i]))
        if short_side(orig) >= 384 and orig.layers == 3:
            img = crop_central_square(orig)
            i_hr = img.resize((256, 256), Image.ANTIALIAS)
            i_lr = img.resize((int(256 / scale), int(256 / scale)))
            name = re.sub('\.jpg$', '', imgs[i])
            name = name.replace('COCO_train2014_', '')
            i_hr.save('data/target/{x}.jpg'.format(x=name), 'JPEG')
            i_lr.save('data/train/{x}.jpg'.format(x=name), 'JPEG')

def square_patch(img_path, size=32):
    img = Image.open(img_path)
    patches = []
    scale = int(img.size[0] / size)
    # Quadratic time, but since it will be used to scale 64x64 images to 32x32 patches, it's viable.
    for i in range(scale):
        for j in range(scale):
            patches.append(img.crop((i * size, j * size, (i + 1) * size, (j + 1) * size)))
    return patches

def train(net, criterion, optimizer, epoch, device, batch_size=16, steps=50):
    net.train()

    losses = []
    data = COCO('data/train/', 'data/target/')
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)
    for e in range(epoch):
        print('Epoch %d' % e)
        for h in range(steps):
            avg_loss = 0
            loss_list, batch_list = [], []

            for i, (images, targets) in enumerate(data_loader):

                optimizer.zero_grad()

                output = net(images.to(device))
                loss = criterion(output, targets.to(device)).cuda()
                avg_loss += loss

                loss_list.append(loss.detach().cuda().item())
                batch_list.append(i + 1)

                losses.append(loss.detach().cuda().item())

                loss.backward()
                optimizer.step()

            print('Epoch: %d - Step: %d, Avg. Loss: %f' % (e, h, avg_loss / batch_size))

        if (e+1) % 50 == 0:
            print('Saving checkpoint.')
            torch.save(net.state_dict(), 'state_{d}e_50s.pth'.format(d=e+1))


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

def resume_training(state_dict_path, net, criterion, optimizer, epoch, device, batch_size=16, steps=50):
    net.train()
    net.load_state_dict(torch.load(state_dict_path))
    losses = []
    data = COCO('data/train/', 'data/target/')
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)
    for e in range(epoch):
        print('Epoch %d' % e)
        for h in range(steps):
            avg_loss = 0
            loss_list, batch_list = [], []

            for i, (images, targets) in enumerate(data_loader):
                optimizer.zero_grad()

                output = net(images.to(device))
                loss = criterion(output, targets.to(device))
                avg_loss += loss

                loss_list.append(loss.detach().cuda().item())
                batch_list.append(i + 1)

                losses.append(loss.detach().cuda().item())

                loss.backward()
                optimizer.step()

            print('Epoch: %d - Step: %d, Avg. Loss: %f' % (e, h, avg_loss / batch_size))

        print('Saving checkpoint.')
        torch.save(net.state_dict(), 'state_{d}e_50s.pth'.format(d=e + 1))

if __name__ == '__main__':
    net = FCNN(input_channels=3)
    net.cuda()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train(net, nn.MSELoss(), optim.Adam(net.parameters(), lr=1e-4), 150, device, batch_size=4, steps=50)





