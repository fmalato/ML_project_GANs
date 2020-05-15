import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from nets import FCNN
from torch.utils.data import DataLoader

from dataset import COCO


def train(net, criterion, optimizer, device, epochs, batch_size=16):
    net.train()

    losses = []
    data = COCO('data/train/', 'data/target/')
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)
    for e in range(epochs):
        print('Epoch %d.' % e)

        for i, (images, targets) in enumerate(data_loader):
            avg_loss = 0
            optimizer.zero_grad()

            output = net(images.to(device))
            loss = criterion(output, targets.to(device)).cuda()
            avg_loss += loss

            losses.append(loss.detach().cuda().item())

            loss.backward()
            optimizer.step()

            print('Step: %d - Avg. Loss: %f' % (i, avg_loss / batch_size))

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
            avg_loss = 0
            optimizer.zero_grad()

            output = net(images.to(device))
            loss = criterion(output, targets.to(device)).cuda()
            avg_loss += loss

            losses.append(loss.detach().cuda().item())

            loss.backward()
            optimizer.step()

            print('Step: %d - Avg. Loss: %f' % (i, avg_loss / batch_size))

        if (e+1) % 5 == 0:
            print('Saving checkpoint.')
            torch.save(net.state_dict(), 'state_{d}e.pth'.format(d=e + starting_epoch + 1))

if __name__ == '__main__':
    net = FCNN(input_channels=3)
    net.cuda()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #resume_training('state_10e.pth', net, nn.MSELoss(), optim.Adam(net.parameters(), lr=1e-5), device, epochs=10, starting_epoch=10, batch_size=64)
    train(net, nn.MSELoss(), optim.Adam(net.parameters(), lr=1e-4), device, epochs=10, batch_size=64)




