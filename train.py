import numpy as np
import time
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch import Tensor
from torch.utils.data import DataLoader

from nets import FCNN, VGGFeatureExtractor, Discriminator
from dataset import COCO
from utils import init_weights, square_patch
from losses import LossE, LossP, LossA, LossT


def multiple_train(net, loss_type, optimizer, device, epochs, batch_size=1, intermediate_step=False):
    net.train()
    data = COCO('data/train/', 'data/target/')
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)
    lossA = False
    losses = []
    losses_d = []
    criterions = []
    for el in loss_type:
        if el == 'E':
            criterions.append(LossE)
        elif el == 'P':
            criterions.append(LossP)
            vgg = [VGGFeatureExtractor(), VGGFeatureExtractor(pool_layer_num=36)]
        elif el == 'A':
            criterions.append(LossA)
            disc = Discriminator()
            disc.cuda()
            optim_d = optim.Adam(disc.parameters(), lr=1e-4)
            lossA = True
        elif el == 'T':
            criterions.append(LossT)
            vgg_T = [VGGFeatureExtractor(pool_layer_num=0),
                     VGGFeatureExtractor(pool_layer_num=5),
                     VGGFeatureExtractor(pool_layer_num=10)]

    for e in range(epochs):
        start = time.perf_counter()
        start_step = start
        print('Epoch %d.' % e)
        epoch_times = []

        for i, (images, targets) in enumerate(data_loader):
            optimizer.zero_grad()

            loss = Tensor(np.zeros(1)).cuda()
            output = net(images.to(device))

            for criterion in criterions:
                if criterion == LossP:
                    loss += criterion(vgg, device, output, targets.to(device))

                elif criterion == LossA:
                    if LossT in criterions:
                        loss_g, loss_d = criterion(net, disc, device, images, targets.to(device), optim_d, True)
                    else:
                        loss_g, loss_d = criterion(net, disc, device, images, targets.to(device), optim_d, False)
                    loss += loss_g

                elif criterion == LossT:
                    loss += criterion(vgg_T, device, output, targets.to(device))

                else:
                    loss += criterion(vgg, device, output, targets.to(device))

            losses.append(loss.detach().cuda().item())

            loss.backward()
            optimizer.step()

            if lossA:
                losses_d.append(loss_d.detach().cuda().item())


            if i % 100 == 0 and i is not 0:
                end_step = time.perf_counter()
                print('Epoch %d - Step: %d    Avg. Loss G: %f    Avg. Loss D: %f' % (e,
                                                                                     i,
                                                                                     sum(losses) / 100,
                                                                                     sum(losses_d) / 100 if lossA else 0.0))
                epoch_times.append(end_step - start_step)
                hours, rem = divmod((sum(epoch_times) / len(epoch_times)) * (int(149400 / batch_size) - i) / 100, 3600)
                minutes, seconds = divmod(rem, 60)
                print('Time for the last step: {:05.2f} s    Epoch ETA: {:0>2}:{:0>2}:{:0>2}'.format(
                    end_step - start_step,
                    int(hours),
                    int(minutes),
                    int(seconds)))
                losses = []
                losses_d = []
                start_step = time.perf_counter()

            if intermediate_step:
                if i % 10000 == 0:
                    print('Saving intermediate checkpoint.')
                    torch.save(net.state_dict(), 'state_{d}e_{x}s_PA.pth')

        end = time.perf_counter()
        print('Epoch %d ended, elapsed time: %f seconds.' % (e, round((end - start), 2)))

    print('Saving checkpoint.')
    torch.save(net.state_dict(), 'state_{d}e_{mode}.pth'.format(d=e+1, mode=''.join(loss_type)))


if __name__ == '__main__':
    batch_size = 16
    net = FCNN(input_channels=3, batch_size=batch_size)
    net.cuda()
    net.apply(init_weights)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    multiple_train(net, ['P', 'A'], optim.Adam(net.parameters(), lr=1e-4), device, epochs=1, batch_size=batch_size, intermediate_step=False)




