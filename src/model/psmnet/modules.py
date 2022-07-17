
from __future__ import print_function

import numpy as np
import torch
import math
import pdb
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from src.module.asm.basics import convbn, convbn_3d


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


# feature extraction module (modified to be light)
class feature_extraction(nn.Module):
    def __init__(self, option, interp='bilinear'):
        super(feature_extraction, self).__init__()
        inplanes = option.inplanes
        inplanes_quad = inplanes // 4
        inplanes_half = inplanes // 2
        inplanes2 = inplanes*2
        inplanes4 = inplanes*4
        inplanes10 = inplanes*10
        self.interp = interp

        self.firstconv = nn.Sequential(convbn(3, inplanes, 3, 2, 1, 1),
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

        # activation function
        self.prelu = nn.ReLU(inplace=True)

        # Weight initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

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

        output = self.firstconv(x)

        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)

        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.interpolate(output_branch1, size=(output_skip.size()[2], output_skip.size()[3]), mode=self.interp, align_corners=True)

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.interpolate(output_branch2, size=(output_skip.size()[2], output_skip.size()[3]), mode=self.interp, align_corners=True)

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.interpolate(output_branch3, size=(output_skip.size()[2], output_skip.size()[3]), mode=self.interp, align_corners=True)

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.interpolate(output_branch4, size=(output_skip.size()[2], output_skip.size()[3]), mode=self.interp, align_corners=True)

        output_feature = torch.cat(
            (output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)

        output_feature = self.lastconv(output_feature)

        return output_feature
    
    
class CostVolume(nn.Module):
    def __init__(self, option, mindisp, maxdisp):
        super(CostVolume, self).__init__()

        self.style = option.model.cost_volume
        self.level = option.model.level

        # for gwcnet style cost volume
        self.group_num = option.model.group_num

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

    def groupwise_correlation(self, fea1, fea2, num_groups):
        B, C, H, W = fea1.shape
        assert C % num_groups == 0
        channels_per_group = C // num_groups
        cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
        assert cost.shape == (B, num_groups, H, W)
        return -cost

    def build_concat_volume(self, refimg_fea, targetimg_fea):
        B, C, H, W = refimg_fea.shape
        volume = torch.zeros([B, 2 * C, int(self.level), H, W], requires_grad=True).cuda()

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

    def build_gwc_volume(self, refimg_fea, targetimg_fea):
        B, C, H, W = refimg_fea.shape
        volume = torch.zeros([B, self.group_num, int(self.level), H, W], requires_grad=True).cuda()

        for i, disp in enumerate(self.costrange):

            disp = int(disp)
            if disp == 0:
                volume[:, :, i, :, :] = self.groupwise_correlation(refimg_fea, targetimg_fea, self.group_num)
            elif disp > 0:
                volume[:, :, i, :-disp, :] = self.groupwise_correlation(refimg_fea[:, :, :-disp, :],
                                                                       targetimg_fea[:, :, disp:, :],
                                                                       self.group_num)
            else:
                volume[:, :, i, -disp:, :] = self.groupwise_correlation(refimg_fea[:, :, -disp:, :],
                                                                       targetimg_fea[:, :, :disp, :],
                                                                       self.group_num)

        volume = volume.contiguous()
        return volume

    def forward(self, ref_feat, tar_feat):

        if self.style == 'psmnet':
            volume = self.build_concat_volume(ref_feat, tar_feat)
        elif self.style == 'gwcnet':
            volume_concat = self.build_concat_volume(ref_feat, tar_feat)
            volume_gwc = self.build_gwc_volume(ref_feat, tar_feat)
            volume = torch.cat((volume_concat, volume_gwc), 1)
        else:
            raise NotImplementedError('cost volume style is not defined : %s' % self.style)

        return volume
    
    
# PSMNet Hourglass network
class PSMNetHourglass(nn.Module):
    def __init__(self, inplanes):
        super(PSMNetHourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes * 2))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes))  # +x

        # Weight initialize
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.ConvTranspose3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post


# if in_channel is less than 8, it might cause exploding of gradient
class PSMNetHGAggregation(nn.Module):
    """22 3D conv"""

    def __init__(self, option):
        super(PSMNetHGAggregation, self).__init__()

        in_channel = option.model.inplanes
        if option.model.cost_volume == 'psmnet':
            first_channel = in_channel * 2
        else:
            first_channel = in_channel * 2 + option.model.group_num
        self.dres0 = nn.Sequential(convbn_3d(first_channel, in_channel, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(in_channel, in_channel, 3, 1, 1),
                                   nn.ReLU(inplace=True))  # [B, 32, D/4, H/4, W/4]

        self.dres1 = nn.Sequential(convbn_3d(in_channel, in_channel, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(in_channel, in_channel, 3, 1, 1))  # [B, 32, D/4, H/4, W/4]

        self.dres2 = PSMNetHourglass(in_channel)

        self.dres3 = PSMNetHourglass(in_channel)

        self.dres4 = PSMNetHourglass(in_channel)

        self.classif1 = nn.Sequential(convbn_3d(in_channel, in_channel, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(in_channel, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(in_channel, in_channel, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(in_channel, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(in_channel, in_channel, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(in_channel, 1, kernel_size=3, padding=1, stride=1, bias=False))

        # Weight initialize
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.ConvTranspose3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, cost):
        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3 + cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        cost3 = F.interpolate(cost3, scale_factor=4, mode='trilinear', align_corners=True)
        cost3 = torch.squeeze(cost3, 1)

        if self.training:
            cost1 = F.interpolate(cost1, scale_factor=4, mode='trilinear', align_corners=True)
            cost2 = F.interpolate(cost2, scale_factor=4, mode='trilinear', align_corners=True)
            cost1 = torch.squeeze(cost1, 1)
            cost2 = torch.squeeze(cost2, 1)
            return [cost3, cost2, cost1]
        else:
            return [cost3], [out3]