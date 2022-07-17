
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


# basic convolution module
def conv3d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=dilation, dilation=dilation,
                                   bias=False, groups=groups),
                         nn.BatchNorm3d(out_channels),
                         nn.LeakyReLU(0.2, inplace=True))


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False)
    bn = nn.BatchNorm2d(out_planes)

    return nn.Sequential(conv, bn)


def tconvbn(in_planes, out_planes, kernel_size, stride):
    
    return nn.Sequential(nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1,
                                            output_padding=1, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False)
    bn = nn.BatchNorm3d(out_planes)

    return nn.Sequential(conv, bn)


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, bias=False, reluw=0.05, bn=True, relu=True):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)
        if bn:
            self.bn = nn.BatchNorm2d(nout)
        if relu:
            self.prelu = nn.PReLU(init=reluw)
        self.use_bn = bn
        self.use_relu = relu

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        if self.use_bn:
            out = self.bn(out)
        if self.use_relu:
            out = self.prelu(out)
        return out
    
    
class BasicBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad=1, deconv=False, is_3d=False, bn=True, relu=True, reluw = 0.05, bias=False, **kwargs):
        super(BasicBlock, self).__init__()
        self.use_relu = relu
        self.use_bn = bn
        if relu:
            self.prelu = nn.PReLU(init=reluw)

        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size,
                                               padding=pad, stride=stride, bias=bias, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                      padding=pad,  stride=stride, bias=bias, **kwargs)
            if bn:
                self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                                               padding=pad, stride=stride, bias=bias, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                               padding=pad, stride=stride, bias=bias, **kwargs)
            if bn:
                self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_relu:
            x = self.prelu(x)
        return x


class DomainNorm(nn.Module):
    def __init__(self, channel, l2=True):
        super(DomainNorm, self).__init__()
        self.normalize = nn.InstanceNorm2d(num_features=channel, affine=False)
        self.l2 = l2
        self.weight = Parameter(torch.ones(1,channel,1,1))
        self.bias = Parameter(torch.zeros(1,channel,1,1))
        self.weight.requires_grad = True
        self.bias.requires_grad = True

    def forward(self, x):
        x = self.normalize(x)
        if self.l2:
            x = F.normalize(x, p=2, dim=1)
        return x * self.weight + self.bias