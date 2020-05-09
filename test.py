import torch
import torchvision.transforms as transforms
import numpy as np

from torch import FloatTensor
from PIL import Image

from nets import FCNN

def test_single(net, image_folder):
    net.eval()
    img = Image.open(image_folder + 'lr.jpg')
    input = FloatTensor(np.array(img))
    input = input.view(-1, 3, 64, 64)
    output = net(input)
    trans = transforms.ToPILImage(mode='RGB')
    output = output.view((3, 256, 256))
    output = trans(output)
    print(output.size)
    target = Image.open(image_folder + 'hr.jpg')
    output.show()
    target.show()


if __name__ == '__main__':

    net = FCNN(input_channels=3)
    net.load_state_dict(torch.load('best_6.pth'))
    test_single(net, 'data/000000000034/')
