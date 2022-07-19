
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.module.asm.basics import convbn, convbn_3d


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride, downsample, pad, dilation):
        super().__init__()
        self.conv1 = nn.Sequential(
            convbn(in_channel, out_channel, 3, stride, pad, dilation),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv2 = convbn(out_channel, out_channel, 3, 1, pad, dilation)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)

        # out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)
        ### bug?
        out = x + out
        return out


class FeatureExtraction(nn.Module):
    def __init__(self, k, in_channel_):
        super().__init__()
        self.k = k
        self.downsample = nn.ModuleList()
        in_channel = in_channel_
        out_channel = 32
        for _ in range(k):
            self.downsample.append(
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size=5,
                    stride=2,
                    padding=2))
            in_channel = out_channel
            out_channel = 32
        self.residual_blocks = nn.ModuleList()
        for _ in range(6):
            self.residual_blocks.append(
                BasicBlock(
                    32, 32, stride=1, downsample=None, pad=1, dilation=1))
        self.conv_alone = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
    def forward(self, rgb_img):
        output = rgb_img
        for i in range(self.k):
            output = self.downsample[i](output)
        for block in self.residual_blocks:
            output = block(output)
        return self.conv_alone(output)
    
    
class EdgeAwareRefinement(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv2d_feature = nn.Sequential(
            convbn(in_channel, 32, kernel_size=3, stride=1, pad=1, dilation=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.residual_astrous_blocks = nn.ModuleList()
        astrous_list = [1, 2, 4, 8 , 1 , 1]
        for di in astrous_list:
            self.residual_astrous_blocks.append(
                BasicBlock(
                    32, 32, stride=1, downsample=None, pad=1, dilation=di))
                
        self.conv2d_out = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, low_disparity, corresponding_rgb):
        output = torch.unsqueeze(low_disparity, dim=1)
        twice_disparity = F.interpolate(
            output,
            size = corresponding_rgb.size()[-2:],
            mode='bilinear',
            align_corners=False)
        if corresponding_rgb.size()[-1]/ low_disparity.size()[-1] >= 1.5:
            twice_disparity *= 8   
        output = self.conv2d_feature(
            torch.cat([twice_disparity, corresponding_rgb], dim=1))
        for astrous_block in self.residual_astrous_blocks:
            output = astrous_block(output)
        
        return nn.ReLU(inplace=True)(torch.squeeze(
            twice_disparity + self.conv2d_out(output), dim=1))
        
        
# disparity regression module
class disp_regression(nn.Module):
    def __init__(self, mindisp, maxdisp, level):
        super(disp_regression, self).__init__()
        multiplier = 1
        self.disparity = np.array(range(int(multiplier * level))) * ((maxdisp - mindisp) / float(multiplier * level)) + mindisp
        self.initialize = False

    def disparity_range_init(self, x):
        self.disp = torch.from_numpy(np.reshape(self.disparity, [1, -1, 1, 1])).type(x.type()).cuda()
        self.initialize = True

    def forward(self, x):
        disparity_out = []
        prob_out = []
        for sample in x:
            assert sample.dim() == 4
            sample = F.softmax(sample, dim=1)
            if not self.initialize:
                self.disparity_range_init(sample)
            disparity_out.append(torch.sum(sample * self.disp, 1, keepdim=False))
            prob_out.append(sample)
        return disparity_out, prob_out
