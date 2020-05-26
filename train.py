import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.models import vgg19

from nets import FCNN
from dataset import COCO
from utils import init_weights
from losses import LossE, LossP


def train(net, criterion, optimizer, device, epochs, batch_size=16):

    net.train()

    losses = []
    data = COCO('data/train/', 'data/target/')
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)
    if criterion == LossP:
        vgg = vgg19(pretrained=True, progress=False)
        vgg.cuda()

    for e in range(epochs):
        start = time.perf_counter()
        print('Epoch %d.' % e)

        for i, (images, targets) in enumerate(data_loader):
            optimizer.zero_grad()

            output = net(images.to(device))
            if criterion == LossP:
                loss = criterion(vgg, device, output, targets.to(device))
            else:
                loss = criterion(device, output, targets.to(device))

            losses.append(loss.detach().cuda().item())

            loss.backward()
            optimizer.step()

            if i % 100 == 0 and i is not 0:
                print('Epoch %d - Step: %d    Avg. Loss: %f' % (e, i, sum(losses) / 100))
                losses = []

        end = time.perf_counter()
        print('Epoch %d ended, elapsed time: %f seconds.' % (e, round((end - start), 2)))

    print('Saving checkpoint.')
    torch.save(net.state_dict(), 'state_{d}e.pth'.format(d=e+1))

def resume_training(state_dict_path, net, criterion, optimizer, device, epochs, starting_epoch, batch_size=16):
    net.train()
    print('Loading state_dict.')
    net.load_state_dict(torch.load(state_dict_path))
    losses = []
    data = COCO('data/train/', 'data/target/')
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)
    print('Resuming training from epoch %d.' % starting_epoch)
    if criterion == LossP:
        vgg = vgg19(pretrained=True)
        vgg.cuda()

    for e in range(epochs):
        start = time.perf_counter()
        print('Epoch %d' % (e + starting_epoch))

        for i, (images, targets) in enumerate(data_loader):
            optimizer.zero_grad()

            output = net(images.to(device))
            if criterion == LossP:
                loss = criterion(vgg, device, output, targets.to(device))
            else:
                loss = criterion(device, output, targets.to(device))

            losses.append(loss.detach().cuda().item())

            loss.backward()
            optimizer.step()

            if i % 100 == 0 and i is not 0:
                print('Epoch %d - Step: %d    Avg. Loss: %f' % (e + starting_epoch, i, sum(losses) / 100))
                losses = []

        end = time.perf_counter()
        print('Epoch %d ended, elapsed time: %f seconds.' % (e, round((end - start), 2)))

    print('Saving checkpoint.')
    torch.save(net.state_dict(), 'state_{d}e.pth'.format(d=e + starting_epoch + 1))

if __name__ == '__main__':
    batch_size = 1
    net = FCNN(input_channels=3, batch_size=batch_size)
    net.cuda()
    net.apply(init_weights)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #resume_training('state_10e_LossE.pth', net, nn.MSELoss(), optim.Adam(net.parameters(), lr=1e-5), device, epochs=1, starting_epoch=10, batch_size=64)
    train(net, LossE, optim.Adam(net.parameters(), lr=1e-4), device, epochs=1, batch_size=batch_size)




