from torch import nn
from collections import OrderedDict


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.blocks = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))),
            ('relu1', nn.ReLU()),
            ('c2', nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)))
            ]))
        self.shortcut = nn.Conv2d(64, 64, kernel_size=(3, 3))

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class FCNN(nn.Module):

    def __init__(self, input_channels=3):
        super().__init__()
        self.input_channels = input_channels

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
        x = self.conv1(x)
        x = self.residual(x)
        x = self.upsamp1(x)
        x = self.upsamp2(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
