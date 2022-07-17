
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from einops import rearrange, repeat

from src.utils.geometry import disp2depth
from src.module.asm.basics import convbn_3d
from src.model.nnet.modules import convtext


class NormalModule(nn.Module):
    
    def __init__(self, option, mindisp, maxdisp):
        super(NormalModule, self).__init__()
        self.option = option
        self.inplanes = option.model.inplanes
        self.mindisp = mindisp
        self.maxdisp = maxdisp
        
        self.wc0 = nn.Sequential(convbn_3d(self.inplanes * 2 + 3, self.inplanes, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(self.inplanes, self.inplanes, 3, 1, 1),
                                     nn.ReLU(inplace=True))
        
        self.pool1 = nn.Sequential(convbn_3d(self.inplanes, self.inplanes, (2,3,3), (2,1,1), (0,1,1)),
                                        nn.ReLU(inplace=True))
        self.pool2 = nn.Sequential(convbn_3d(self.inplanes, self.inplanes, (2,3,3), (2,1,1), (0,1,1)),
                                        nn.ReLU(inplace=True))
        self.pool3 = nn.Sequential(convbn_3d(self.inplanes, self.inplanes, (2,3,3), (2,1,1), (0,1,1)),
                                        nn.ReLU(inplace=True))
        self.n_convs = nn.Sequential(
            convtext(self.inplanes, self.inplanes * 3, 3, 1, 1),
            convtext(self.inplanes * 3, self.inplanes * 3, 3, 1, 2),
            convtext(self.inplanes * 3, self.inplanes * 3, 3, 1, 4),
            convtext(self.inplanes * 3, self.inplanes * 2, 3, 1, 8),
            convtext(self.inplanes * 2, self.inplanes * 2, 3, 1, 16),
            convtext(self.inplanes * 2, self.inplanes, 3, 1, 1),
            convtext(self.inplanes, 3, 3, 1, 1)
        )
        
        # registered params
        self.grid_check = False
        costrange = torch.arange(option.model.level) * ((self.maxdisp / 4.0 - self.mindisp / 4.0) / 
                                                         float(option.model.level)) + self.mindisp / 4.0
        self.register_parameter('costrange', Parameter(costrange.view(1, -1, 1, 1), False))

    def grid_maker_3d(self, cost, K, disp_map, ab_value=None):
        '''
        Make 3D world coordinate volume
        :param cost: [B, D, C, H, W]
        :param disp_map: [B, 1, H, W]
        :param ab_value: [B, 2]
        :return:
        '''
        
        b, d, c, h, w = cost.shape

        if self.grid_check is False:
            x = torch.arange(0, w).type_as(K)
            y = torch.arange(0, h).type_as(K)
            y_grid, x_grid = torch.meshgrid([y, x])
            ones_matrix = torch.ones_like(x_grid)
            grid = rearrange([x_grid, y_grid, ones_matrix], 'c h w -> c h w')
            grid = Parameter(grid.unsqueeze(0), False)  # [1, 3, H, W]
            self.register_parameter('grid', grid)
            self.grid_check = True

        # K-1 * [u, v, 1]
        K_imgF = K.clone()
        K_imgF[:, :2, :] = K_imgF[:, :2, :] / 4.0
        warp_grid = repeat(self.grid, 'b c h w -> (repeat b) c (h w)', repeat=b)
        warp_grid = torch.bmm(torch.inverse(K_imgF), warp_grid).view(b, 3, h, w)  # [B, 3, H, W]
        warp_grid = warp_grid.type_as(disp_map)

        # disparity to depth conversion
        depth = disp2depth(disp_map, ab_value)
        warp_grid_3d = rearrange(warp_grid, 'b c h w -> b c 1 h w') * rearrange(depth, 'b d h w -> b 1 d h w')  # [B, 3, D, H, W]
        
        # normalize volume (scale normalize)
        warp_grid_min = rearrange(torch.min(warp_grid_3d.view(b, -1), -1)[0], 'b -> b 1 1 1 1')
        warp_grid_max = rearrange(torch.max(warp_grid_3d.view(b, -1), -1)[0], 'b -> b 1 1 1 1')
        nwarp_grid_3d = (warp_grid_3d - warp_grid_min) / (warp_grid_max - warp_grid_min + 1e-6)

        return nwarp_grid_3d.contiguous()

    def forward(self, cost_in, batch):
        '''
        :param cost_in: [B, C, D, H, W]
        :return:
        '''
        
        # params
        gt_depth = batch['depth'] if 'depth' in batch.keys() else None  # not used
        ab_value = batch['abvalue'] if 'abvalue' in batch.keys() else None
        
        b,ch,d,h,w = cost_in.size()
        
        # Coordinate Volume
        disp_range = self.costrange.expand(b, -1, h, w).type_as(cost_in)  # [B, D, H, W]
        wc = self.grid_maker_3d(cost_in, batch['K'], disp_range, ab_value)
        
        # apply normal layers
        wc = torch.cat((wc.clone(), cost_in), dim = 1).contiguous()
        wc0 = self.pool3(self.pool2(self.pool1(self.wc0(wc))))
        
        slices = []
        nmap = torch.zeros((b,3,h,w)).type_as(wc0)
        for i in range(wc0.size(2)):
            slices.append(self.n_convs(wc0[:,:,i]))
            nmap += slices[-1]
            
        nmap = F.interpolate(nmap, scale_factor=4, mode='bilinear', align_corners=True)
        nmap = F.normalize(nmap, dim=1)

        return [nmap]