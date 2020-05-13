import torch
import torchvision.transforms as transforms
import numpy as np
import os

from torch import FloatTensor
from PIL import Image

from nets import FCNN
from train import load_data

def test_single(net, image_folder):
    net.eval()
    img = Image.open(image_folder + 'train/000000000034.jpg')
    tens = transforms.ToTensor()
    input = tens(img)
    input = input.view((1, 3, 64, 64))
    output = net(input)
    trans = transforms.ToPILImage(mode='RGB')
    output = output.view((3, 256, 256))
    output = trans(output)
    output.show()


if __name__ == '__main__':

    net = FCNN(input_channels=3)
    img = Image.open('data/train/000000000034.jpg')
    target = Image.open('data/target/000000000034.jpg')
    net.load_state_dict(torch.load('state_1800e_50s.pth', map_location=torch.device('cpu')))
    test_single(net, 'data/')
    net.load_state_dict(torch.load('state_300e_50s_1batch_1e-4.pth', map_location=torch.device('cpu')))
    test_single(net, 'data/')
    net.load_state_dict(torch.load('state_800e_50s.pth', map_location=torch.device('cpu')))
    test_single(net, 'data/')
    img.show()
    target.show()


