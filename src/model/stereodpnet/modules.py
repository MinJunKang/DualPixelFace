
from __future__ import print_function
from collections import OrderedDict

import pdb
import math
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F


from torchvision.ops import FeaturePyramidNetwork
from src.module.asm.basics import convbn, convbn_3d, depthwise_separable_conv
from src.module.asm.asm import subpixel_shift, MaskingAttention
    
    
# o = [i + 2p - k - (k-1)(d-1)] / s + 1
class DPBlock(nn.Module):

    def __init__(self, inplanes, ratio_s, ratio_t, reluw=0.05):
        super(DPBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, inplanes, 3, 1, 1, 1), nn.PReLU(init=reluw))
        self.conv2 = nn.Sequential(convbn(inplanes, inplanes, 3, 1, 1, 1), nn.PReLU(init=reluw))
        self.conv_dilate = nn.ModuleList([convbn(inplanes, inplanes, 3, 1, 2 * i + 1, 2 * i + 1) for i in range(3)])
        self.conv3 = convbn(inplanes * 3, inplanes, 3, 1, 1, 1)
        self.conv4 = nn.Sequential(convbn(inplanes, ratio_t*inplanes, 3, ratio_s, ratio_s, 2), nn.PReLU(init=reluw))
        self.conv5 = depthwise_separable_conv(ratio_t * inplanes, ratio_t * inplanes, 3, 1)
        self.conv_skip = nn.Conv2d(inplanes, ratio_t*inplanes, 1, ratio_s)

        # activation function
        self.prelu = nn.PReLU(init=reluw)

    def forward(self, x):

        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        
        out2 = torch.cat([self.conv_dilate[0](out2), self.conv_dilate[1](out2), self.conv_dilate[2](out2)], dim=1)

        out2 = self.conv3(out2)
        out = self.prelu(out2 + out1)  # skip connection

        out = self.conv4(out)
        out = self.conv5(out)

        out = out + self.conv_skip(x)  # weighted skip connection

        return out
    
    
# feature extraction module (modified to be light)
class feature_extraction(nn.Module):

    def __init__(self, option, reluw=0.05):
        super(feature_extraction, self).__init__()
        self.inplanes = option.model.inplanes
        self.blockstack = option.model.block_stack

        # half the resolution
        self.firstconv = nn.Sequential(convbn(option.model.input_channel, self.inplanes, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(self.inplanes, self.inplanes, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(self.inplanes, self.inplanes, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        # layer1 : 1/4 resolution
        self.block1 = DPBlock(self.inplanes, 2, 1)

        # layer2 : 1/8 resolution
        self.interblock1 = nn.ModuleList([DPBlock(self.inplanes, 1, 1) for i in range(self.blockstack)])
        self.block2 = DPBlock(self.inplanes, 2, 2)

        # layer3 : 1/16 resolution
        self.interblock2 = nn.ModuleList([DPBlock(self.inplanes*2, 1, 1) for i in range(self.blockstack)])
        self.block3 = DPBlock(self.inplanes*2, 2, 2)

        # FPN Multiscale Fusion
        self.fpn = FeaturePyramidNetwork(in_channels_list=[self.inplanes, self.inplanes*2, self.inplanes*4],
                                         out_channels=self.inplanes,
                                         extra_blocks=None)

        # Last Conv
        self.lastconv = nn.Sequential(convbn(self.inplanes * 3, self.inplanes * 2, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      convbn(self.inplanes * 2, self.inplanes, 3, 1, 1, 1), 
                                      nn.ReLU(inplace=True))

    def forward(self, x):

        # Initial conv
        initial = self.firstconv(x)

        # Encoder 1
        out1 = self.block1(initial)

        # Encoder 2
        out2 = out1
        for i in range(self.blockstack):
            out2 = self.interblock1[i](out2)
        out2 = self.block2(out2)

        # Encoder 3
        out3 = out2
        for i in range(self.blockstack):
            out3 = self.interblock2[i](out3)
        out3 = self.block3(out3)

        # FPN
        fpn_input = OrderedDict()
        fpn_input['layer1'] = out1  # [C, H/4, W/4]
        fpn_input['layer2'] = out2  # [2C, H/8, W/8]
        fpn_input['layer3'] = out3  # [4C, H/16, W/16]

        fpn_output = self.fpn(fpn_input)

        out_high = fpn_output['layer1']
        out_mid = fpn_output['layer2']
        out_low = fpn_output['layer3']

        # output feature
        stage0 = out_high
        stage1 = F.interpolate(out_mid, scale_factor=2, mode='bilinear', align_corners=True)
        stage2 = F.interpolate(out_low, scale_factor=4, mode='bilinear', align_corners=True)
        output_feature = torch.cat([stage0, stage1, stage2], dim=1)

        # High level feature
        output_feature = self.lastconv(output_feature)

        return output_feature
    
    
class CostVolume(nn.Module):
    def __init__(self, option, mindisp, maxdisp):
        super(CostVolume, self).__init__()

        self.level = option.model.level

        # cost volume range
        self.costrange = np.array(range(int(self.level))) * (
                (maxdisp / 4.0 - mindisp / 4.0) / float(self.level)) + mindisp / 4.0

        # Adaptive Sampling Module
        self.shifting_layer = subpixel_shift(option)
        self.attention_layer = MaskingAttention(option.model.inplanes, 
                                                act=option.model.asm_activation, 
                                                feature_fetch=option.model.feature_fetch)

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

            feat_forward = self.shifting_layer(srcfeat=refimg_fea, disp=disp, dir='forward')
            feat_backward = self.shifting_layer(srcfeat=targetimg_fea, disp=disp, dir='backward')

            feat_forward = torch.cat(feat_forward, -1).permute(0, 1, 4, 2, 3)
            feat_backward = torch.cat(feat_backward, -1).permute(0, 1, 4, 2, 3)

            volume[:, :C, i, :, :] = self.attention_layer(feat_forward)
            volume[:, C:, i, :, :] = self.attention_layer(feat_backward)

        volume = volume.contiguous()
        return volume

    def forward(self, ref_feat, tar_feat):
        return self.build_concat_volume(ref_feat, tar_feat)
    
    
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

    def __init__(self, in_channel):
        super(PSMNetHGAggregation, self).__init__()
        self.multiplier = 4

        self.dres0 = nn.Sequential(convbn_3d(in_channel * 2, in_channel, 3, 1, 1),
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

        cost3 = F.interpolate(cost3, scale_factor=self.multiplier, mode='trilinear', align_corners=True)
        cost3 = torch.squeeze(cost3, 1)

        if self.training:
            cost1 = F.interpolate(cost1, scale_factor=self.multiplier, mode='trilinear', align_corners=True)
            cost2 = F.interpolate(cost2, scale_factor=self.multiplier, mode='trilinear', align_corners=True)
            cost1 = torch.squeeze(cost1, 1)
            cost2 = torch.squeeze(cost2, 1)
            return [cost3, cost2, cost1], [out3, out2, out1]
        else:
            return [cost3], [out3]
        
        
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
        
        
