import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from nets import FCNN
from dataset import COCO
from utils import init_weights


def train(net, criterion, optimizer, device, epochs, batch_size=16):
    net.train()

    losses = []
    data = COCO('data/train/', 'data/target/')
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)
    for e in range(epochs):
        print('Epoch %d.' % e)

        for i, (images, targets) in enumerate(data_loader):
            optimizer.zero_grad()

            output = net(images.to(device))
            loss = criterion(output, targets.to(device)).cuda()

            losses.append(loss.detach().cuda().item())

            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print('Epoch %d - Step: %d    Loss: %f' % (e, i, loss))

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
    for e in range(epochs):
        print('Epoch %d' % (e + starting_epoch))

        for i, (images, targets) in enumerate(data_loader):
            optimizer.zero_grad()

            output = net(images.to(device))
            loss = criterion(output, targets.to(device)).cuda()

            losses.append(loss.detach().cuda().item())

            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print('Epoch %d - Step: %d    Loss: %f' % (e + starting_epoch, i, loss))


    print('Saving checkpoint.')
    torch.save(net.state_dict(), 'state_{d}e.pth'.format(d=e + starting_epoch + 1))

if __name__ == '__main__':
    net = FCNN(input_channels=3)
    net.cuda()
    net.apply(init_weights)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #resume_training('state_10e.pth', net, nn.MSELoss(), optim.Adam(net.parameters(), lr=1e-5), device, epochs=10, starting_epoch=10, batch_size=64)
    train(net, nn.MSELoss(), optim.Adam(net.parameters(), lr=1e-4), device, epochs=10, batch_size=1)




