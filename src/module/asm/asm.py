import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# grid sampling module #
class subpixel_shift(nn.Module):

    def __init__(self, option):
        super(subpixel_shift, self).__init__()
        self.mode_nearest = option.model.nearest
        self.mode_bilinear = option.model.bilinear
        self.mode_phase = option.model.phase
        self.basic_grid_forward = None
        self.basic_grid_backward = None
        self.phase_grid_forward = None
        self.phase_grid_backward = None

    def make_grid(self, h, w, disp, dir, mode):

        if dir == 'forward':
            sign = 1.0
        else:
            sign = -1.0

        if mode == 'basic':
            if ((dir == 'forward') and (self.basic_grid_forward is None)) or (
                    (dir == 'backward') and (self.basic_grid_backward is None)):

                deltar = torch.tensor(float(sign * disp)).cuda()
                deltac = torch.tensor(0.0).cuda()

                y_range = (torch.arange(0.0, h).cuda() + deltar)
                x_range = (torch.arange(0.0, w).cuda() + deltac)
                yv, xv = torch.meshgrid([y_range, x_range])

                # normalize
                xv = xv / (w - 1) * 2.0 - 1.0
                yv = yv / (h - 1) * 2.0 - 1.0
                xv = xv.unsqueeze(0)
                yv = yv.unsqueeze(0)

                # grid sampling
                if dir == 'forward':
                    self.basic_grid_forward = torch.cat([xv, yv], dim=0).permute([1, 2, 0])
                else:
                    self.basic_grid_backward = torch.cat([xv, yv], dim=0).permute([1, 2, 0])

            if dir == 'forward':
                return self.basic_grid_forward
            else:
                return self.basic_grid_backward
        else:
            if ((dir == 'forward') and (self.phase_grid_forward is None)) or (
                    (dir == 'backward') and (self.phase_grid_backward is None)):

                deltar = torch.tensor(float(sign * disp)).cuda()
                deltac = torch.tensor(0.0).cuda()

                # convert to device tensor
                deltar = torch.tensor(deltar / h, requires_grad=False).cuda()
                deltac = torch.tensor(deltac / w, requires_grad=False).cuda()

                # range
                Nr = torch.tensor(np.concatenate([torch.arange(0.0, np.ceil(h // 2)), torch.arange(-np.fix(h // 2), 0.0)]),
                                  requires_grad=False).cuda()
                Nc = torch.tensor(np.concatenate([torch.arange(0.0, np.ceil(w // 2)), torch.arange(-np.fix(w // 2), 0.0)]),
                                  requires_grad=False).cuda()
                Nr, Nc = torch.meshgrid([Nr, Nc])

                # cosine term
                costerm = torch.cos(torch.tensor(2.0 * np.pi).cuda() * (deltar * Nr + deltac * Nc))
                sinterm = torch.sin(torch.tensor(2.0 * np.pi).cuda() * (deltar * Nr + deltac * Nc))

                if dir == 'forward':
                    self.phase_grid_forward = [costerm, sinterm]
                else:
                    self.phase_grid_backward = [costerm, sinterm]

            if dir == 'forward':
                return self.phase_grid_forward
            else:
                return self.phase_grid_backward

    def forward(self, srcfeat, disp, dir):

        dstfeats = []
        batch_size, channel, h, w = srcfeat.shape

        if self.mode_nearest:

            grid = self.make_grid(h, w, disp, dir, 'basic').expand(batch_size, -1, -1, -1)

            dstfeats.append(F.grid_sample(srcfeat, grid=grid.type_as(srcfeat), mode='nearest').unsqueeze(-1))

        if self.mode_bilinear:
            grid = self.make_grid(h, w, disp, dir, 'basic').expand(batch_size, -1, -1, -1)

            dstfeats.append(F.grid_sample(srcfeat, grid=grid.type_as(srcfeat), mode='bilinear',
                                          align_corners=True).unsqueeze(-1))

        if self.mode_phase:
            grid = self.make_grid(h, w, disp, dir, 'phase')
            costerm = grid[0].repeat(batch_size, channel, 1, 1)
            sinterm = grid[1].repeat(batch_size, channel, 1, 1)

            # Apply fft to src
            # src : [batch, channel, height, width]
            # output : [batch, channel, height, width, real/complex]
            value = torch.rfft(srcfeat.float(), 2, onesided=False)

            fr = value[:, :, :, :, 0]
            fi = value[:, :, :, :, 1]

            # Shifting in fourier domain
            fr2 = fr * costerm - fi * sinterm
            fi2 = fi * costerm + fr * sinterm

            # inverse fft
            fr2 = fr2.unsqueeze(-1)
            fi2 = fi2.unsqueeze(-1)

            dstfeats.append(torch.irfft(torch.cat([fr2, fi2], -1), 2, onesided=False).type_as(srcfeat).unsqueeze(-1))

        return dstfeats


# Adaptive Sampling Module
class MaskingAttention(nn.Module):
    def __init__(self, nin, bias=False, act='relu', feature_fetch=False):
        super(MaskingAttention, self).__init__()
        kernel_size = 3
        padsize1 = (kernel_size - 1) // 2

        # last normalization layer
        self.normalize = nn.InstanceNorm3d(nin, affine=True)

        # mask layers
        self.mask_convs = nn.Sequential(nn.Conv3d(nin, nin, kernel_size=(1, kernel_size, kernel_size), stride=1,
                               padding=(0, padsize1, padsize1), bias=bias),
                               nn.BatchNorm3d(nin),
                               nn.ReLU(inplace=True),
                               nn.Sequential(nn.Conv3d(nin, nin, kernel_size=1, stride=1, padding=0, bias=bias),
                               self.normalize))

        self.feature_fetch = feature_fetch

        # activation layer
        if act == 'relu':
            self.activation = nn.PReLU(init=0.05)
        elif act == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise NotImplementedError('activation type is not implemented')

    def forward(self, x):

        mask = self.mask_convs(x)
        
        x = x * F.softmax(self.activation(mask), dim=2)

        # followed from feature fetching of MVSNet, get variance feature
        if self.feature_fetch:
            avg_x = torch.mean(x, 2, keepdim=False)
            avg_x_2 = torch.mean(x ** 2, 2, keepdim=False)

            output = avg_x_2 - (avg_x ** 2)
        else:
            output = torch.mean(x, 2, keepdim=False)

        return output
    
    
