import numpy as np
import time
import re
import torch
import torch.optim as optim

from torch import Tensor
from torch.utils.data import DataLoader
from datetime import date

from nets import FCNN, VGGFeatureExtractor, Discriminator
from dataset import COCO
from utils import init_weights
from losses import LossE, LossP, LossA, LossT, LossA_2


def multiple_train(net, loss_type, optimizer, device, epochs, batch_size=1, load_weights=False, state_dict=''):
    net.train()
    step_update = 100
    data = COCO('data/train/', 'data/target/')
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)
    lossA = False
    losses = []
    losses_d = []
    losses_g = []
    D_xs = []
    D_gs = []
    criterions = []
    D_x = D_G_z = 0.0
    num_imgs = len(data_loader)
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
            criterions.append(LossA_2)
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
        print('Epoch %d.' % (e + 1))
        epoch_times = []

        for i, (images, targets, bicub) in enumerate(data_loader):
            optimizer.zero_grad()

            images.to(device)
            targets.to(device)
            bicub.to(device)

            loss = Tensor(np.zeros(1))
            output = net(images.float())
            output = torch.add(output, bicub).clamp(0, 1)
            output.to(device)

            for criterion in criterions:
                if criterion == LossP:
                    loss += criterion(vgg, device, output.float(), targets.float())

                elif criterion == LossA_2:
                    if 'T' in loss_type:
                        loss_g, loss_d, D_x, D_G_z = criterion(disc, device, output.float(),
                                                               targets.float(), optim_d, D_x, D_G_z,
                                                               True)
                    else:
                        loss_g, loss_d, D_x, D_G_z = criterion(disc, device, output.float(),
                                                               targets.float(), optim_d, D_x, D_G_z,
                                                               False)
                    loss += loss_g

                elif criterion == LossT:
                    loss += criterion(vgg_T, device, output.float(), targets.float())

                else:
                    loss += criterion(device, output.float(), targets.float())

            losses.append(loss.detach().item())

            loss.backward()
            optimizer.step()

            if lossA:
                losses_d.append(loss_d.detach().item())
                losses_g.append(loss_g.detach().item())
                D_xs.append(D_x)
                D_gs.append(D_G_z)

            if i % step_update == 0 and i is not 0:
                end_step = time.perf_counter()
                print('Epoch %d/%d - Step: %d/%d  Loss G: %f (%f) Loss D: %f  D(x): %f  D(G(z)): %f' % (e + 1, epochs,
                                                                                                        i, len(data_loader),
                                                                                                        sum(losses) / step_update,
                                                                                                        sum(losses_g) / step_update if lossA else 0.0,
                                                                                                        sum(losses_d) / step_update if lossA else 0.0,
                                                                                                        sum(D_xs) / step_update if lossA else 0.0,
                                                                                                        sum(D_gs) / step_update if lossA else 0.0))
                epoch_times.append(end_step - start_step)
                hours, rem = divmod((sum(epoch_times) / len(epoch_times)) * (num_imgs - i) / 100, 3600)
                minutes, seconds = divmod(rem, 60)
                print('Time for the last step: {:05.2f} s    Epoch ETA: {:0>2}:{:0>2}:{:0>2}'.format(
                    end_step - start_step,
                    int(hours),
                    int(minutes),
                    int(seconds)))
                losses = []
                losses_d = []
                losses_g = []
                D_xs = []
                D_gs = []
                start_step = time.perf_counter()

        end = time.perf_counter()
        print('Epoch %d ended, elapsed time: %f seconds.' % (e + 1, round((end - start), 2)))

    print('Saving checkpoint.')
    today = date.today()
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    if load_weights:
        torch.save(net.state_dict(), 'state_{d}e_{mode}_{date}_{time}.pth'.format(d=e + starting_epoch + 1,
                                                                                  mode=''.join(loss_type),
                                                                                  date=today.strftime("%b-%d-%Y"),
                                                                                  time=current_time))
    else:
        torch.save(net.state_dict(), 'state_{d}e_{mode}_{date}_{time}.pth'.format(d=e + 1,
                                                                                  mode=''.join(loss_type),
                                                                                  date=today.strftime("%b-%d-%Y"),
                                                                                  time=current_time))


if __name__ == '__main__':
    batch_size = 8
    epochs = 1
    lr = 1e-4
    loss_type = ['P', 'A']
    load_weights = False
    state_dict = 'state_1e_E'
    net = FCNN(input_channels=3, batch_size=batch_size)
    net.float()
    net.cuda()
    net.apply(init_weights)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    try:
        multiple_train(net, loss_type, optim.Adam(net.parameters(), lr=lr), device, epochs=epochs,
                       batch_size=batch_size * 4,
                       load_weights=load_weights, state_dict=state_dict)
    except KeyboardInterrupt:
        print('Training interrupted. Saving model.')
        today = date.today()
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        torch.save(net.state_dict(), 'state_interrupt_{mode}_{date}_{time}.pth'.format(mode=''.join(loss_type),
                                                                                       date=today.strftime("%b-%d-%Y"),
                                                                                       time=current_time))
