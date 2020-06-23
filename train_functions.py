import numpy as np
import time
import re
import torch
import torch.optim as optim

from torch import Tensor
from torch.utils.data import DataLoader
from datetime import date
from torch.autograd import Variable

from nets import FCNN, VGGFeatureExtractor, Discriminator
from dataset import COCO
from utils import init_weights, print_stats, time_stats, true_or_false, compute_patches
from losses import LossE, LossP, LossA, LossT


def trainE(net, disc, optim_g, optim_d, device, data_loader, start_step, current_epoch, epochs=1, train_disc=True,
           step_update=100, batch_size=1):

    losses = []
    epoch_times = []

    for i, (images, targets, bicub) in enumerate(data_loader):
        optim_g.zero_grad()

        images = images.to(device)
        targets = targets.to(device)
        bicub = bicub.to(device)
        images = images.view((-1, 3, 32, 32))
        targets = targets.view((-1, 3, 128, 128))
        bicub = bicub.view((-1, 3, 128, 128))

        output = net(images.float())
        output = torch.add(output, bicub).clamp(0, 1)
        output = output.to(device)

        loss = LossE(device, output.float(), targets.float())

        losses.append(loss.detach().item())

        loss.backward()
        optim_g.step()

        if i % step_update == 0 and i is not 0:
            end_step = time.perf_counter()
            epoch_times.append(end_step - start_step)
            print_stats(False, current_epoch, epochs, len(data_loader), i, losses, [], [], [], [], step_update)
            time_stats(epoch_times, end_step, start_step, len(data_loader), i)
            losses = []
            start_step = time.perf_counter()



def trainP(net, disc, optim_g, optim_d, device, data_loader, start_step, current_epoch, epochs=1, train_disc=True,
           step_update=100, batch_size=1):

    losses = []
    epoch_times = []
    vgg = [VGGFeatureExtractor().float().cuda(), VGGFeatureExtractor(pool_layer_num=36).float().cuda()]

    for i, (images, targets, bicub) in enumerate(data_loader):
        optim_g.zero_grad()

        images = images.to(device)
        targets = targets.to(device)
        bicub = bicub.to(device)
        images = images.view((-1, 3, 32, 32))
        targets = targets.view((-1, 3, 128, 128))
        bicub = bicub.view((-1, 3, 128, 128))

        images = images.squeeze(0)
        targets = targets.squeeze(0)
        bicub = bicub.squeeze(0)

        output = net(images.float())
        output = torch.add(output, bicub).clamp(0, 1)
        output = output.to(device)

        loss = LossP(device, vgg[0](output.float()), vgg[0](targets.float()),
                      vgg[1](output.float()), vgg[1](targets.float()))

        losses.append(loss.detach().item())

        loss.backward()
        optim_g.step()

        if i % step_update == 0 and i is not 0:
            end_step = time.perf_counter()
            epoch_times.append(end_step - start_step)
            print_stats(False, current_epoch, epochs, len(data_loader), i, losses, [], [], [], [], step_update)
            time_stats(epoch_times, end_step, start_step, len(data_loader), i)
            losses = []
            start_step = time.perf_counter()


def trainEA(net, disc, optim_g, optim_d, device, data_loader, start_step, current_epoch, epochs=1, train_disc=True,
            step_update=100, batch_size=1):

    losses = []
    losses_d = []
    losses_g = []
    D_xs = []
    D_gs = []
    epoch_times = []

    for i, (images, targets, bicub) in enumerate(data_loader):
        optim_g.zero_grad()

        images = images.to(device)
        targets = targets.to(device)
        bicub = bicub.to(device)
        images = images.view((-1, 3, 32, 32))
        targets = targets.view((-1, 3, 128, 128))
        bicub = bicub.view((-1, 3, 128, 128))

        loss = Tensor(np.zeros(1)).cuda()
        output = net(images.float())
        output = torch.add(output, bicub).clamp(0, 1)
        output = output.to(device)

        # discriminator, device, output_g, target, optim_d, last_batch, lossT=False, train_disc=False
        loss += LossE(device, output.float(), targets.float())
        loss_g, loss_d, D_x, D_G_z = LossA(disc, device, output.float(), targets.float(), optim_d,
                                               True, train_disc=train_disc)
        loss += loss_g.mean().item()

        losses.append(loss.detach().item())
        losses_d.append(loss_d.detach().mean().item())
        losses_g.append(loss_g.detach().mean().item())
        D_xs.append(D_x)
        D_gs.append(D_G_z)

        loss.backward()
        optim_g.step()

        with torch.no_grad():
            d_true = true_or_false(disc(targets.float()).cpu().detach().numpy())
            d_fake = true_or_false(disc(output.float()).cpu().detach().numpy())
            perf_true = d_true.count(1) / len(d_true)
            perf_fake = d_fake.count(0) / len(d_fake)
            if perf_fake < 0.8 or perf_true < 0.8:
                train_disc = True
            else:
                train_disc = False

        conv = check_convergence(D_xs, D_gs)
        # Reaching convergence after just 3 batches sounds like an unfortunate event, not convergence
        if conv and i > int(1000 / (current_epoch+1)):
            today = date.today()
            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            torch.save(net.state_dict(), 'state_converged_{date}_{time}.pth'.format(date=today.strftime("%b-%d-%Y"),
                                                                                    time=current_time))
            raise KeyboardInterrupt('Convergence reached.')
            
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


def trainPA(net, disc, optim_g, optim_d, device, data_loader, start_step, current_epoch, epochs=1, train_disc=True,
            step_update=100, batch_size=1):

    losses = []
    losses_d = []
    losses_g = []
    D_xs = []
    D_gs = []
    epoch_times = []
    vgg = [VGGFeatureExtractor().float(), VGGFeatureExtractor(pool_layer_num=36).float()]

    for i, (images, targets, bicub) in enumerate(data_loader):
        optim_g.zero_grad()

        images = images.to(device)
        targets = targets.to(device)
        bicub = bicub.to(device)
        images = images.view((-1, 3, 32, 32))
        targets = targets.view((-1, 3, 128, 128))
        bicub = bicub.view((-1, 3, 128, 128))

        loss = Tensor(np.zeros(1)).cuda()
        output = net(images.float())
        output = torch.add(output, bicub).clamp(0, 1)
        output = output.to(device)

        loss += LossP(device, vgg[0](output.float()), vgg[0](targets.float()),
                      vgg[1](output.float()), vgg[1](targets.float()))
        loss_g, loss_d, D_x, D_G_z = LossA(disc, device, output.float(), targets.float(), optim_d,
                                           True, train_disc=train_disc)
        loss += loss_g.mean().item()

        losses.append(loss.detach().item())
        losses_d.append(loss_d.detach().mean().item())
        losses_g.append(loss_g.detach().mean().item())
        D_xs.append(D_x)
        D_gs.append(D_G_z)

        loss.backward()
        optim_g.step()

        with torch.no_grad():
            d_true = true_or_false(disc(targets.float()).cpu().detach().numpy())
            d_fake = true_or_false(disc(output.float()).cpu().detach().numpy())
            perf_true = d_true.count(1) / len(d_true)
            perf_fake = d_fake.count(0) / len(d_fake)
            if perf_fake < 0.8 or perf_true < 0.8:
                train_disc = True
            else:
                train_disc = False

        conv = check_convergence(D_xs, D_gs)
        # Reaching convergence after just 3 batches sounds like an unfortunate event, not convergence
        if conv and i > int(1000 / (current_epoch + 1)):
            today = date.today()
            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            torch.save(net.state_dict(), 'state_converged_{date}_{time}.pth'.format(date=today.strftime("%b-%d-%Y"),
                                                                                    time=current_time))

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


def trainEAT(net, disc, optim_g, optim_d, device, data_loader, start_step, current_epoch, epochs=1, train_disc=True,
             step_update=100, batch_size=1):
    losses = []
    losses_d = []
    losses_g = []
    D_xs = []
    D_gs = []
    epoch_times = []
    vgg_T = [VGGFeatureExtractor(pool_layer_num=0).float().cuda(),
             VGGFeatureExtractor(pool_layer_num=5).float().cuda(),
             VGGFeatureExtractor(pool_layer_num=10).float().cuda()
             ]

    for i, (images, targets, bicub) in enumerate(data_loader):
        optim_g.zero_grad()

        images = images.to(device)
        targets = targets.to(device)
        bicub = bicub.to(device)
        images = images.view((-1, 3, 32, 32))
        targets = targets.view((-1, 3, 128, 128))
        targets = Variable(targets.type(torch.cuda.FloatTensor))
        bicub = bicub.view((-1, 3, 128, 128))

        loss = Variable(Tensor(np.zeros(1)), requires_grad=True)
        loss_t = Variable(Tensor(np.zeros(1)), requires_grad=True)
        output = net(images.float())
        output = torch.add(output, bicub).clamp(0, 1)
        output = Variable(output.type(torch.cuda.FloatTensor))

        loss = loss + Variable(LossE(device, output.float(), targets.float()), requires_grad=True)
        loss_g, loss_d, D_x, D_G_z = LossA(disc, device, output.float(), targets.float(), optim_d,
                                           True, train_disc=train_disc)
        loss = loss + Variable(Tensor(loss_g), requires_grad=True)
        patches, patches_target = compute_patches(output, targets)
        idx = 0
        for im, trg in zip(patches, patches_target):
            # One patch every 16 is enough, otherwise it will slow down computation too much
            if idx % 16 == 0:
                loss_t = loss_t + Variable(3e-7 * LossT(device, vgg_T[0](im.float()), vgg_T[0](trg.float())), requires_grad=True)
                loss_t = loss_t + Variable(1e-6 * LossT(device, vgg_T[1](im.float()), vgg_T[1](trg.float())), requires_grad=True)
                loss_t = loss_t + Variable(1e-6 * LossT(device, vgg_T[2](im.float()), vgg_T[2](trg.float())), requires_Grad=True)
            idx += 1
        loss = loss + Variable((loss_t / len(patches)).to(device), requires_grad=True)

        losses.append(loss.detach().item())
        losses_d.append(loss_d.detach().mean().item())
        losses_g.append(loss_g.detach().mean().item())
        D_xs.append(D_x)
        D_gs.append(D_G_z)

        loss.backward()
        optim_g.step()

        with torch.no_grad():
            d_true = true_or_false(disc(targets.float()).cpu().detach().numpy())
            d_fake = true_or_false(disc(output.float()).cpu().detach().numpy())
            perf_true = d_true.count(1) / len(d_true)
            perf_fake = d_fake.count(0) / len(d_fake)
            if perf_fake < 0.8 or perf_true < 0.8:
                train_disc = True
            else:
                train_disc = False

        conv = check_convergence(D_xs, D_gs)
        # Reaching convergence after just 3 batches sounds like an unfortunate event, not convergence
        if conv and i > int(1000 / (current_epoch + 1)):
            today = date.today()
            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            torch.save(net.state_dict(), 'state_converged_{date}_{time}.pth'.format(date=today.strftime("%b-%d-%Y"),
                                                                                    time=current_time))
            

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


def trainPAT(net, disc, optim_g, optim_d, device, data_loader, start_step, current_epoch, epochs=1, train_disc=True,
             step_update=100, batch_size=1):

    losses = []
    losses_d = []
    losses_g = []
    D_xs = []
    D_gs = []
    epoch_times = []
    vgg = [VGGFeatureExtractor().float().cuda(), VGGFeatureExtractor(pool_layer_num=36).float().cuda()]
    vgg_T = [VGGFeatureExtractor(pool_layer_num=0).float().cuda(),
             VGGFeatureExtractor(pool_layer_num=5).float().cuda(),
             VGGFeatureExtractor(pool_layer_num=10).float().cuda()]

    for i, (images, targets, bicub) in enumerate(data_loader):
        optim_g.zero_grad()
        
        images = images.to(device)
        targets = targets.to(device)
        bicub = bicub.to(device)
        images = images.view((-1, 3, 32, 32))
        targets = targets.view((-1, 3, 128, 128))
        bicub = bicub.view((-1, 3, 128, 128))

        loss = Tensor(np.zeros(1)).cuda()
        loss_t = Tensor(np.zeros(1)).cuda()
        output = net(images.float())
        output = torch.add(output, bicub).clamp(0, 1)
        output = output.to(device)

        loss += LossP(device, vgg[0](output.float()), vgg[0](targets.float()),
                      vgg[1](output.float()), vgg[1](targets.float()))
        loss_g, loss_d, D_x, D_G_z = LossA(disc, device, output.float(), targets.float(), optim_d,
                                           True, train_disc=train_disc)
        loss += loss_g.mean().item()
        patches, patches_target = compute_patches(output, targets, step=16)
        for im, trg in zip(patches, patches_target):
                loss_t += 3e-7 * LossT(device, vgg_T[0](im.float()), vgg_T[0](trg.float()))
                loss_t += 1e-6 * LossT(device, vgg_T[1](im.float()), vgg_T[1](trg.float()))
                loss_t += 1e-6 * LossT(device, vgg_T[2](im.float()), vgg_T[2](trg.float()))
        loss += (loss_t / len(patches))

        losses.append(loss.detach().item())
        losses_d.append(loss_d.detach().mean().item())
        losses_g.append(loss_g.detach().mean().item())
        D_xs.append(D_x)
        D_gs.append(D_G_z)

        loss.backward()
        optim_g.step()

        with torch.no_grad():
            d_true = true_or_false(disc(targets.float()).cpu().detach().numpy())
            d_fake = true_or_false(disc(output.float()).cpu().detach().numpy())
            perf_true = d_true.count(1) / len(d_true)
            perf_fake = d_fake.count(0) / len(d_fake)
            if perf_fake < 0.8 or perf_true < 0.8:
                train_disc = True
            else:
                train_disc = False

        conv = check_convergence(D_xs, D_gs)
        # Reaching convergence after just 3 batches sounds like an unfortunate event, not convergence
        if conv and i > int(1000 / (current_epoch + 1)):
            today = date.today()
            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            torch.save(net.state_dict(), 'state_converged_{date}_{time}.pth'.format(date=today.strftime("%b-%d-%Y"),
                                                                                    time=current_time))

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

def generate_means(batch_size, size=32, scale_factor=4):
    PER_CHANNEL_MEANS_32 = np.zeros((batch_size, 3, size, size))
    for i in range(batch_size):
        PER_CHANNEL_MEANS_32[i][0].fill(0.47614917)
        PER_CHANNEL_MEANS_32[i][1].fill(0.45001204)
        PER_CHANNEL_MEANS_32[i][2].fill(0.40904046)
    PER_CHANNEL_MEANS_32 = torch.from_numpy(PER_CHANNEL_MEANS_32)
    PER_CHANNEL_MEANS_128 = np.zeros((batch_size, 3, size * scale_factor, size * scale_factor))
    for i in range(batch_size):
        PER_CHANNEL_MEANS_128[i][0].fill(0.47614917)
        PER_CHANNEL_MEANS_128[i][1].fill(0.45001204)
        PER_CHANNEL_MEANS_128[i][2].fill(0.40904046)
    PER_CHANNEL_MEANS_128 = torch.from_numpy(PER_CHANNEL_MEANS_128)

    return PER_CHANNEL_MEANS_32, PER_CHANNEL_MEANS_128

def check_convergence(dx, dg):
    x_mean = sum(dx) / len(dx)
    g_mean = sum(dg) / len(dg)
    if (0.45 < x_mean < 0.55) and (0.45 < g_mean < 0.55):
        return True

    return False
