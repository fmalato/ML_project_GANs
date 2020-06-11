import numpy as np
import time
import os, re
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


def multiple_train(net, loss_type, optimizer, device, epochs, batch_size=1, intermediate_step=False, load_weights=False, state_dict=''):
    net.train()
    data = COCO('data/train/', 'data/target/')
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)
    lossA = False
    losses = []
    losses_d = []
    criterions = []
    D_x = D_G_z1 = D_G_z2 = 0.0
    if load_weights:
        print('Loading {x}'.format(x=state_dict))
        net.load_state_dict(torch.load('trained_models/{x}.pth'.format(x=state_dict), map_location=torch.device('cpu')))
        starting_epoch = int(re.sub("[^0-9]", "", state_dict))
    for el in loss_type:
        if el == 'E':
            criterions.append(LossE)
        elif el == 'P':
            criterions.append(LossP)
            vgg = [VGGFeatureExtractor().float(), VGGFeatureExtractor(pool_layer_num=36).float()]
        elif el == 'A':
            criterions.append(LossA)
            disc = Discriminator()
            disc.float()
            disc.cuda()
            optim_d = optim.Adam(disc.parameters(), lr=1e-4)
            lossA = True
        elif el == 'T':
            criterions.append(LossT)
            vgg_T = [VGGFeatureExtractor(pool_layer_num=0).float(),
                     VGGFeatureExtractor(pool_layer_num=5).float(),
                     VGGFeatureExtractor(pool_layer_num=10).float()]

    for e in range(epochs):
        start = time.perf_counter()
        start_step = start
        print('Epoch %d.' % (e+1))
        epoch_times = []

        for i, (images, targets, bicub) in enumerate(data_loader):
            optimizer.zero_grad()
            images = images.view((-1, images.shape[2], images.shape[3], images.shape[4]))
            targets = targets.view((-1, targets.shape[2], targets.shape[3], targets.shape[4]))
            bicub = bicub.view((-1, bicub.shape[2], bicub.shape[3], bicub.shape[4]))
            images.cuda()
            targets.cuda()
            bicub.cuda()

            loss = Tensor(np.zeros(1)).cuda()
            output = net(images.float(), bicub.float())
            output = torch.add(output, bicub)

            for criterion in criterions:
                if criterion == LossP:
                    loss += criterion(vgg, device, output.float(), targets.float())

                elif criterion == LossA:
                    if LossT in criterions:
                        loss_g, loss_d, D_x, D_G_z1, D_G_z2 = criterion(net, disc, device, images.float(), targets.float(), bicub.float(), optim_d, True)
                    else:
                        loss_g, loss_d, D_x, D_G_z1, D_G_z2 = criterion(net, disc, device, images.float(), targets.float(), bicub.float(), optim_d, False)
                    loss += loss_g

                elif criterion == LossT:
                    loss += criterion(vgg_T, device, output.float(), targets.float())

                else:
                    loss += criterion(device, output.float(), targets.float())

            losses.append(loss.detach().cuda().item())

            loss.backward()
            optimizer.step()

            if lossA:
                losses_d.append(loss_d.detach().cuda().item())

            if i % 100 == 0 and i is not 0:
                end_step = time.perf_counter()
                print('Epoch %d/%d - Step: %d/%d  Loss G: %f  Loss D: %f  D(x): %f  D(G(z)): %f / %f' % (e+1, epochs,
                                                                                                         i, len(data_loader),
                                                                                                         sum(losses) / 100,
                                                                                                         sum(losses_d) / 100 if lossA else 0.0,
                                                                                                         D_x,
                                                                                                         D_G_z1,
                                                                                                         D_G_z2))
                epoch_times.append(end_step - start_step)
                hours, rem = divmod((sum(epoch_times) / len(epoch_times)) * (int(232013 / batch_size) - i) / 100, 3600)
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
        print('Epoch %d ended, elapsed time: %f seconds.' % (e+1, round((end - start), 2)))

    print('Saving checkpoint.')
    if load_weights:
        torch.save(net.state_dict(), 'state_{d}e_{mode}.pth'.format(d=e+starting_epoch+1, mode=''.join(loss_type)))
    else:
        torch.save(net.state_dict(), 'state_{d}e_{mode}.pth'.format(d=e+1, mode=''.join(loss_type)))


if __name__ == '__main__':
    batch_size = 48
    epochs = 3
    lr = 1e-4
    loss_type = ['E']
    load_weights = False
    state_dict = ''
    net = FCNN(input_channels=3, batch_size=batch_size)
    net.float()
    net.cuda()
    net.apply(init_weights)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    try:
        multiple_train(net, loss_type, optim.Adam(net.parameters(), lr=lr), device, epochs=epochs, batch_size=batch_size,
                       intermediate_step=False, load_weights=load_weights, state_dict=state_dict)
    except KeyboardInterrupt:
        print('Training interrupted. Saving model.')
        torch.save(net.state_dict(), 'state_int_{x}e_{mode}.pth'.format(x=epochs, mode=''.join(loss_type)))





