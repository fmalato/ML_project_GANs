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
from utils import init_weights, print_stats, time_stats
from losses import LossE, LossP, LossA, LossT


def trainE(net, disc, optim_g, optim_d, device, data_loader, start_step, current_epoch, epochs=1, first_step=True, step_update=100):

    losses = []
    epoch_times = []

    for i, (images, targets, bicub) in enumerate(data_loader):
        optim_g.zero_grad()

        images = images.to(device)
        targets = targets.to(device)
        bicub = bicub.to(device)

        loss = Tensor(np.zeros(1)).cuda()
        output = net(images.float())
        output = torch.add(output, bicub).clamp(0, 1)
        output = output.to(device)

        loss += LossE(device, output.float(), targets.float())

        losses.append(loss.detach().item())

        loss.backward()
        optim_g.step()

        if i % step_update == 0 and i is not 0:
            end_step = time.perf_counter()
            epoch_times.append(end_step - start_step)
            print_stats(False, current_epoch, epochs, len(data_loader), i, losses, [], [], [], [], step_update)
            time_stats(epoch_times, end_step, start_step, len(data_loader),i)
            losses = []
            start_step = time.perf_counter()



def trainP(net, disc, optim_g, optim_d, device, data_loader, start_step, current_epoch, epochs=1, first_step=True, step_update=100):

    losses = []
    epoch_times = []
    vgg = [VGGFeatureExtractor().float(), VGGFeatureExtractor(pool_layer_num=36).float()]

    for i, (images, targets, bicub) in enumerate(data_loader):
        optim_g.zero_grad()

        images = images.to(device)
        targets = targets.to(device)
        bicub = bicub.to(device)

        loss = Tensor(np.zeros(1)).cuda()
        output = net(images.float())
        output = torch.add(output, bicub).clamp(0, 1)
        output = output.to(device)

        loss += LossP(vgg, device, output.float(), targets.float())

        losses.append(loss.detach().item())

        loss.backward()
        optim_g.step()

        if i % step_update == 0 and i is not 0:
            end_step = time.perf_counter()
            epoch_times.append(end_step - start_step)
            print_stats(False, current_epoch, epochs, len(data_loader), i, losses, [], [], [], [], step_update)
            time_stats(epoch_times, end_step, start_step, len(data_loader),i)
            losses = []
            start_step = time.perf_counter()


def trainEA(net, disc, optim_g, optim_d, device, data_loader, start_step, current_epoch, epochs=1, first_step=True, step_update=100):

    losses = []
    losses_d = []
    losses_g = []
    last_out = []
    last_tar = []
    D_xs = []
    D_gs = []
    epoch_times = []

    for i, (images, targets, bicub) in enumerate(data_loader):
        optim_g.zero_grad()

        images = images.to(device)
        targets = targets.to(device)
        bicub = bicub.to(device)

        loss = Tensor(np.zeros(1)).cuda()
        output = net(images.float())
        output = torch.add(output, bicub).clamp(0, 1)
        output = output.to(device)

        # discriminator, device, output_g, target, optim_d, last_batch, lossT=False, first_step=False
        loss += LossE(device, output.float(), targets.float())
        loss_g, loss_d, D_x, D_G_z = LossA(disc, device, output.float(), targets.float(), optim_d, [last_out, last_tar],
                                           lossT=False, first_step=first_step)
        loss += loss_g.mean().item()
        first_step = False

        losses.append(loss.detach().item())
        losses_d.append(loss_d.detach().mean().item())
        losses_g.append(loss_g.detach().mean().item())
        D_xs.append(D_x)
        D_gs.append(D_G_z)

        loss.backward()
        optim_g.step()

        if not first_step:
            last_out = output
            last_tar = targets

        if i % step_update == 0 and i is not 0:
            end_step = time.perf_counter()
            epoch_times.append(end_step - start_step)
            print_stats(True, current_epoch, epochs, len(data_loader), i, losses, losses_g, losses_d, D_xs, D_gs,
                        step_update)
            time_stats(epoch_times, end_step, start_step, len(data_loader), i)
            losses = []
            losses_d = []
            losses_g = []
            D_xs = []
            D_gs = []
            start_step = time.perf_counter()


def trainPA(net, disc, optim_g, optim_d, device, data_loader, start_step, current_epoch, epochs=1, first_step=True, step_update=100):

    losses = []
    losses_d = []
    losses_g = []
    last_out = []
    last_tar = []
    D_xs = []
    D_gs = []
    epoch_times = []
    vgg = [VGGFeatureExtractor().float(), VGGFeatureExtractor(pool_layer_num=36).float()]

    for i, (images, targets, bicub) in enumerate(data_loader):
        optim_g.zero_grad()

        images = images.to(device)
        targets = targets.to(device)
        bicub = bicub.to(device)

        loss = Tensor(np.zeros(1)).cuda()
        output = net(images.float())
        output = torch.add(output, bicub).clamp(0, 1)
        output = output.to(device)

        loss += LossP(vgg, device, output.float(), targets.float())
        loss_g, loss_d, D_x, D_G_z = LossA(disc, device, output.float(), targets.float(), optim_d, [last_out, last_tar],
                                           lossT=False, first_step=first_step)
        loss += loss_g.mean().item()

        losses.append(loss.detach().item())
        losses_d.append(loss_d.detach().mean().item())
        losses_g.append(loss_g.detach().mean().item())
        D_xs.append(D_x)
        D_gs.append(D_G_z)

        loss.backward()
        optim_g.step()

        if not first_step:
            last_out = output
            last_tar = targets

        if i % step_update == 0 and i is not 0:
            end_step = time.perf_counter()
            epoch_times.append(end_step - start_step)
            print_stats(True, current_epoch, epochs, len(data_loader), i, losses, losses_g, losses_d, D_xs, D_gs,
                        step_update)
            time_stats(epoch_times, end_step, start_step, len(data_loader), i)
            losses = []
            losses_d = []
            losses_g = []
            D_xs = []
            D_gs = []
            start_step = time.perf_counter()


def trainEAT(net, disc, optim_g, optim_d, device, data_loader, start_step, current_epoch, epochs=1, first_step=True, step_update=100):

    losses = []
    losses_d = []
    losses_g = []
    last_out = []
    last_tar = []
    D_xs = []
    D_gs = []
    epoch_times = []
    vgg_T = [VGGFeatureExtractor(pool_layer_num=0).float(),
             VGGFeatureExtractor(pool_layer_num=5).float(),
             VGGFeatureExtractor(pool_layer_num=10).float()]

    for i, (images, targets, bicub) in enumerate(data_loader):
        optim_g.zero_grad()

        images = images.to(device)
        targets = targets.to(device)
        bicub = bicub.to(device)

        loss = Tensor(np.zeros(1)).cuda()
        output = net(images.float())
        output = torch.add(output, bicub).clamp(0, 1)
        output = output.to(device)

        loss += LossE(device, output.float(), targets.float())
        loss_g, loss_d, D_x, D_G_z = LossA(disc, device, output.float(), targets.float(), optim_d, [last_out, last_tar],
                                           lossT=False, first_step=first_step)
        loss += loss_g.mean().item()
        loss += LossT(vgg_T, device, output.float(), targets.float())

        losses.append(loss.detach().item())
        losses_d.append(loss_d.detach().mean().item())
        losses_g.append(loss_g.detach().mean().item())
        D_xs.append(D_x)
        D_gs.append(D_G_z)

        loss.backward()
        optim_g.step()

        if not first_step:
            last_out = output
            last_tar = targets

        if i % step_update == 0 and i is not 0:
            end_step = time.perf_counter()
            epoch_times.append(end_step - start_step)
            print_stats(True, current_epoch, epochs, len(data_loader), i, losses, losses_g, losses_d, D_xs, D_gs,
                        step_update)
            time_stats(epoch_times, end_step, start_step, len(data_loader), i)
            losses = []
            losses_d = []
            losses_g = []
            D_xs = []
            D_gs = []
            start_step = time.perf_counter()


def trainPAT(net, disc, optim_g, optim_d, device, data_loader, start_step, current_epoch, epochs=1, first_step=True, step_update=100):

    losses = []
    losses_d = []
    losses_g = []
    last_out = []
    last_tar = []
    D_xs = []
    D_gs = []
    epoch_times = []
    vgg = [VGGFeatureExtractor().float(), VGGFeatureExtractor(pool_layer_num=36).float()]
    vgg_T = [VGGFeatureExtractor(pool_layer_num=0).float(),
             VGGFeatureExtractor(pool_layer_num=5).float(),
             VGGFeatureExtractor(pool_layer_num=10).float()]

    for i, (images, targets, bicub) in enumerate(data_loader):
        optim_g.zero_grad()

        images = images.to(device)
        targets = targets.to(device)
        bicub = bicub.to(device)

        loss = Tensor(np.zeros(1)).cuda()
        output = net(images.float())
        output = torch.add(output, bicub).clamp(0, 1)
        output = output.to(device)

        loss += LossP(vgg, device, output.float(), targets.float())
        loss_g, loss_d, D_x, D_G_z = LossA(disc, device, output.float(), targets.float(), optim_d, [last_out, last_tar],
                                           lossT=False, first_step=first_step)
        loss += loss_g.mean().item()
        loss += LossT(vgg_T, device, output.float(), targets.float())

        losses.append(loss.detach().item())
        losses_d.append(loss_d.detach().mean().item())
        losses_g.append(loss_g.detach().mean().item())
        D_xs.append(D_x)
        D_gs.append(D_G_z)

        loss.backward()
        optim_g.step()

        if not first_step:
            last_out = output
            last_tar = targets

        if i % step_update == 0 and i is not 0:
            end_step = time.perf_counter()
            epoch_times.append(end_step - start_step)
            print_stats(True, current_epoch, epochs, len(data_loader), i, losses, losses_g, losses_d, D_xs, D_gs,
                        step_update)
            time_stats(epoch_times, end_step, start_step, len(data_loader), i)
            losses = []
            losses_d = []
            losses_g = []
            D_xs = []
            D_gs = []
            start_step = time.perf_counter()


def which_train(loss_type):
    if loss_type == 'E':
        train = trainE
        is_adv = False
    elif loss_type == 'P':
        train = trainP
        is_adv = False
    elif loss_type == 'EA':
        train = trainEA
        is_adv = True
    elif loss_type == 'PA':
        train = trainPA
        is_adv = True
    elif loss_type == 'EAT':
        train = trainEAT
        is_adv = True
    else:
        train = trainPAT
        is_adv = True
    return train, is_adv
