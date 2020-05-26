import torchvision.transforms as transforms
import torch.nn.functional as F

from torch import nn
from collections import OrderedDict
from PIL import Image

from utils import custom_bicubic


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.block1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))),
            ('relu1', nn.ReLU()),
            ]))
        self.block2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x):
        residual = x
        x = self.block1(x)
        x = self.block2(x)
        x += residual
        return x


class FCNN(nn.Module):

    def __init__(self, input_channels=3, batch_size=1, scale_factor=4):
        super().__init__()
        self.input_channels = input_channels
        self.bicubic_upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.tens = transforms.ToTensor()
        self.pilimg = transforms.ToPILImage()
        self.scale_factor = scale_factor
        self.batch_size = batch_size

        self.conv1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(self.input_channels, 64, kernel_size=(3, 3), padding=(1, 1))),
            ('relu1', nn.ReLU())
        ]))
        self.residual = nn.Sequential(OrderedDict([
            ('res1', ResidualBlock(64, 64)),
            ('res2', ResidualBlock(64, 64)),
            ('res3', ResidualBlock(64, 64)),
            ('res4', ResidualBlock(64, 64)),
            ('res5', ResidualBlock(64, 64)),
            ('res6', ResidualBlock(64, 64)),
            ('res7', ResidualBlock(64, 64)),
            ('res8', ResidualBlock(64, 64)),
            ('res9', ResidualBlock(64, 64)),
            ('res10', ResidualBlock(64, 64))
        ]))
        self.upsamp1 = nn.Sequential(OrderedDict([
            ('up1ctrans', nn.UpsamplingNearest2d(scale_factor=2)),
            ('up1conv', nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))),
            ('up1relu', nn.ReLU())
        ]))
        self.upsamp2 = nn.Sequential(OrderedDict([
            ('up2ctrans', nn.UpsamplingNearest2d(scale_factor=2)),
            ('up2conv', nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))),
            ('up2relu', nn.ReLU())
        ]))
        self.conv2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))),
            ('relu2', nn.ReLU())
        ]))
        self.conv3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(64, 3, kernel_size=(3, 3), padding=(1, 1)))
        ]))

    def forward(self, x):
        y = self.conv1(x)
        y = self.residual(y)
        y = self.upsamp1(y)
        y = self.upsamp2(y)

        y = self.conv2(y)
        y = self.conv3(y)

        return y + custom_bicubic(x.cpu(), self.tens, self.pilimg, self.scale_factor).cuda()



class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(OrderedDict([
            # Block 1
            ('conv1', nn.Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1))),
            ('lReLU1', nn.LeakyReLU()),
            ('conv1b', nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))),
            ('lReLU1b', nn.LeakyReLU()),
            # Block 2
            ('conv2', nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))),
            ('lReLU2', nn.LeakyReLU()),
            ('conv2b', nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))),
            ('lReLU2b', nn.LeakyReLU()),
            # Block 3
            ('conv3', nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))),
            ('lReLU3', nn.LeakyReLU()),
            ('conv3b', nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))),
            ('lReLU3b', nn.LeakyReLU()),
            # Block 4
            ('conv4', nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1))),
            ('lReLU4', nn.LeakyReLU()),
            ('conv4b', nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))),
            ('lReLU4b', nn.LeakyReLU()),
            # Block 5
            ('conv5', nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1))),
            ('lReLU5', nn.LeakyReLU()),
            ('conv5b', nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))),
            ('lReLU5b', nn.LeakyReLU())
            ]))

        self.fc = nn.Sequential(OrderedDict([
            # FC 1
            ('fc1', nn.Linear(8192, 1024)),
            ('lReLUfc1', nn.LeakyReLU()),
            # FC 2
            ('fc2', nn.Linear(1024, 1)),
            ('sigfc2', nn.Sigmoid())
        ]))

    def forward(self, x):
        # Convolution
        x = self.conv(x)
        # Flattening
        x = x.view(x.size(0), -1)
        # FC
        x = self.fc(x)

        return x
