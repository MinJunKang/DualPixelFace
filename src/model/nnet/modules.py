from __future__ import print_function

import math
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from src.module.asm.basics import convbn


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        return out
    
    
def convtext(in_planes, out_planes, kernel_size = 3, stride = 1, dilation = 1):
    
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = stride, dilation = dilation, padding = ((kernel_size - 1) * dilation) // 2, bias = False),
        nn.LeakyReLU(0.1,inplace=True)
    )
    
    
class feature_extraction(nn.Module):
    def __init__(self, option):
        super(feature_extraction, self).__init__()
        inplanes = option.model.inplanes
        inplanes_quad = inplanes // 4
        inplanes_half = inplanes // 2
        inplanes2 = inplanes*2
        inplanes4 = inplanes*4
        inplanes10 = inplanes*10
        
        self.firstconv = nn.Sequential(convbn(option.model.input_channel, inplanes, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(inplanes, inplanes, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(inplanes, inplanes, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, inplanes, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, inplanes2, inplanes_half, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, inplanes4, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, inplanes4, 3, 1, 1, 2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((inplanes2, inplanes2), stride=(inplanes2, inplanes2)),
                                     convbn(inplanes4, inplanes, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((inplanes, inplanes), stride=(inplanes, inplanes)),
                                     convbn(inplanes4, inplanes, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((inplanes_half, inplanes_half),
                                                  stride=(inplanes_half, inplanes_half)),
                                     convbn(inplanes4, inplanes, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((inplanes_quad, inplanes_quad), stride=(inplanes_quad, inplanes_quad)),
                                     convbn(inplanes4, inplanes, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(inplanes10, inplanes4, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inplanes4, inplanes, kernel_size=1, padding=0, stride=1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        # from psm-net
        output      = self.firstconv(x)
        output      = self.layer1(output)
        output_raw  = self.layer2(output)
        output      = self.layer3(output_raw)
        output_skip = self.layer4(output)

        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.interpolate(output_branch1, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear', align_corners = False)

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.interpolate(output_branch2, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear', align_corners = False)

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.interpolate(output_branch3, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear', align_corners = False)

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.interpolate(output_branch4, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear', align_corners = False)

        output_feature = torch.cat((output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature
    
    
class CostVolume(nn.Module):
    def __init__(self, option, mindisp, maxdisp):
        super(CostVolume, self).__init__()
        self.level = option.model.level
        
        # cost volume range
        self.costrange = np.array(range(int(self.level))) * (
                (maxdisp / 4.0 - mindisp / 4.0) / float(self.level)) + mindisp / 4.0

        '''
                (y axis disparity)

                1) disp > 0 case

                reference : 1 1 1 2 3 4 1 1 1 1
                    target : 1 1 1 1 1 2 3 4 1 1

                :-disp = :-disp  (reference)
                :-disp = disp:  (target)

                for disp = 1 case
                reference : 1 1 1 2 3 4 1 1 1 0
                    target : 1 1 1 1 2 3 4 1 1 0

                2) disp < 0 case

                reference : 1 1 1 1 1 2 3 4 1 1
                    target : 1 1 1 2 3 4 1 1 1 1

                -disp: = -disp:  (reference)
                -disp: = :disp  (target)

                for disp = -1 case
                reference : 0 1 1 1 1 2 3 4 1 1
                    target : 0 1 1 1 2 3 4 1 1 1
        '''

    def build_concat_volume(self, refimg_fea, targetimg_fea):
        B, C, H, W = refimg_fea.shape
        volume = torch.zeros([B, 2 * C, int(self.level), H, W], requires_grad=True).type_as(refimg_fea)

        for i, disp in enumerate(self.costrange):

            disp = int(disp)

            if disp == 0:
                volume[:, :C, i, :, :] = refimg_fea
                volume[:, C:, i, :, :] = targetimg_fea
            elif disp > 0:
                volume[:, :C, i, :-disp, :] = refimg_fea[:, :, :-disp, :]
                volume[:, C:, i, :-disp, :] = targetimg_fea[:, :, disp:, :]
            else:
                volume[:, :C, i, -disp:, :] = refimg_fea[:, :, -disp:, :]
                volume[:, C:, i, -disp:, :] = targetimg_fea[:, :, :disp, :]

        volume = volume.contiguous()
        return volume

    def forward(self, ref_feat, tar_feat):
        return self.build_concat_volume(ref_feat, tar_feat)
    
    
# disparity regression module
class disp_regression(nn.Module):
    def __init__(self, mindisp, maxdisp, level):
        super(disp_regression, self).__init__()
        multiplier = 4
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

