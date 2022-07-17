import torch
import torch.nn as nn
import torch.nn.functional as F
from src.module.asm.basics import BasicBlock, depthwise_separable_conv


class Encoder(nn.Module):
    
    def __init__(self, inchannel, infilter, outfilter, stride, pad_basic, reluw = 0.05):
        super(Encoder, self).__init__()

        self.conv1 = nn.Sequential(BasicBlock(inchannel, infilter, kernel_size=3, stride=stride, pad=pad_basic),  # 0
                                   depthwise_separable_conv(infilter, infilter, kernel_size=3, padding=1)  # 1
                                   )

        self.conv2 = BasicBlock(infilter, outfilter, kernel_size=1, stride=1, pad=0)

        self.skip_connection = nn.Sequential(
            BasicBlock(inchannel, outfilter, kernel_size=1, stride=1, pad=pad_basic),
            nn.MaxPool2d(kernel_size=3, stride=stride, padding=0)  # 0
        )

        self.prelu = nn.PReLU(init=reluw)

    def forward(self,x):

        x2 = self.conv1(x)

        x_skip = self.skip_connection(x)
        x2 = self.conv2(x2)

        x2 = x2 + x_skip

        x2 = self.prelu(x2)

        return x2


class Encoder2(nn.Module):

    def __init__(self, inchannel, outfilter, stride):
        super(Encoder2, self).__init__()

        self.conv1 = BasicBlock(inchannel, outfilter, kernel_size=7, stride=stride, pad=1)

        self.maxpool = nn.MaxPool2d(kernel_size=7, stride=stride, padding=1)

    def forward(self, x):

        x_skip = self.maxpool(x)

        x = self.conv1(x)

        x = torch.cat((x, x_skip), dim=1)

        return x


class Decoder(nn.Module):

    def __init__(self, inchannel, infilter, pad_basic, pad_1, pad_2, pad_3, mode=None):
        super(Decoder, self).__init__()

        self.mode = mode

        if mode is None:
            self.conv1 = nn.Sequential(BasicBlock(inchannel, infilter, kernel_size=4, stride=2, deconv=True, pad=pad_basic),
                                       depthwise_separable_conv(infilter, infilter, kernel_size=3, padding=pad_1),
                                       depthwise_separable_conv(infilter, infilter, kernel_size=1, padding=pad_2),
                                       depthwise_separable_conv(infilter, infilter, kernel_size=3, padding=pad_3)
                                       )
        else:
            self.conv1 = nn.Sequential(BasicBlock(inchannel, infilter, kernel_size=3 + 2 * (pad_basic - 1), stride=1, pad=1),
                                       depthwise_separable_conv(infilter, infilter, kernel_size=3, padding=pad_1),
                                       depthwise_separable_conv(infilter, infilter, kernel_size=1, padding=pad_2),
                                       depthwise_separable_conv(infilter, infilter, kernel_size=3, padding=pad_3)
                                       )

    def forward(self, x):

        if self.mode is not None:
            x = F.interpolate(x, scale_factor=2, mode=self.mode)

        return self.conv1(x)


class Decoder2(nn.Module):

    def __init__(self, inchannel, infilter, outfilter, pad_basic, pad_1, pad_2, pad_3, mode=None):
        super(Decoder2, self).__init__()

        self.mode = mode

        if mode is None:
            self.conv1 = nn.Sequential(BasicBlock(inchannel, infilter, kernel_size=4, stride=2, deconv=True, pad=pad_basic),
                                       depthwise_separable_conv(infilter, infilter, kernel_size=3, padding=pad_1),
                                       depthwise_separable_conv(infilter, infilter, kernel_size=1, padding=pad_2),
                                       depthwise_separable_conv(infilter, infilter, kernel_size=3, padding=pad_3),
                                       BasicBlock(infilter, outfilter, kernel_size=1, pad=1, bn=False, relu=False)
                                       )
        else:
            self.conv1 = nn.Sequential(BasicBlock(inchannel, infilter, kernel_size=3 + 2 * (pad_basic - 1), stride=1, pad=1),
                                       depthwise_separable_conv(infilter, infilter, kernel_size=3, padding=pad_1),
                                       depthwise_separable_conv(infilter, infilter, kernel_size=1, padding=pad_2),
                                       depthwise_separable_conv(infilter, infilter, kernel_size=3, padding=pad_3),
                                       BasicBlock(infilter, outfilter, kernel_size=1, pad=1, bn=False, relu=False)
                                       )

    def forward(self, x):

        if self.mode is not None:
            x = F.interpolate(x, scale_factor=2, mode=self.mode)

        return self.conv1(x)