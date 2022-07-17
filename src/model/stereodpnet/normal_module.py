
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from einops import rearrange, repeat
from src.utils.geometry import disp2depth
from src.module.dcn3d.modules import DeformConvPack_dv2
from src.module.asm.basics import convbn_3d


def convtext(in_planes, out_planes, kernel_size = 3, stride = 1, dilation = 1):
    
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = stride, dilation = dilation, padding = ((kernel_size - 1) * dilation) // 2, bias = False),
        nn.LeakyReLU(0.1,inplace=True)
    )
    
    
class upsampler(nn.Module):
    
    def __init__(self, factor=2):
        super(upsampler, self).__init__()
        self.factor = factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.factor, mode='bilinear', align_corners=True)
    
    
class ANM(nn.Module):
    def __init__(self, option, mindisp, maxdisp):
        super(ANM, self).__init__()
        self.option = option
        self.inplanes = option.model.inplanes
        self.mindisp = mindisp
        self.maxdisp = maxdisp
        
        # grid for surface sampling
        self.idx_start = int((option.model.dsample_num - 1) / 2)
        self.idx_end = int(option.model.level) - int(option.model.dsample_num / 2) - 1
        
        # 3D deformable conv (D3D Module)
        if option.model.use_deform:
            self.deform_conv1 = DeformConvPack_dv2(self.inplanes + 3,  self.inplanes * 2,
                                                   kernel_size=[3, 3, 3], stride=1, padding=1, maxsize=self.maxdisp * 2)
            self.act1 = nn.Sequential(nn.BatchNorm3d(self.inplanes * 2), nn.ReLU(inplace=True))
            self.deform_conv2 = DeformConvPack_dv2(self.inplanes * 2, self.inplanes * 2,
                                                   kernel_size=[3, 3, 3], stride=1, padding=1, maxsize=self.maxdisp * 2)
            self.act2 = nn.Sequential(nn.BatchNorm3d(self.inplanes * 2), nn.ReLU(inplace=True))
        else:
            self.original_conv = nn.Sequential(convbn_3d(self.inplanes + 3, self.inplanes * 2, 3, 1, 1),
                                               nn.ReLU(inplace=True),
                                               convbn_3d(self.inplanes * 2, self.inplanes * 2, 3, 1, 1),
                                               nn.ReLU(inplace=True))
            
        # shared normal conv : NNet
        self.n_convs = nn.Sequential(
            convtext(self.inplanes * 2, self.inplanes * 3, 3, 1, 1),
            convtext(self.inplanes * 3, self.inplanes * 3, 3, 1, 2),
            convtext(self.inplanes * 3, self.inplanes * 2, 3, 1, 4),
            convtext(self.inplanes * 2, self.inplanes * 2, 3, 1, 8),
            convtext(self.inplanes * 2, self.inplanes, 3, 1, 1),
            convtext(self.inplanes, 3, 3, 1, 1)
        )
        
        # final layer
        self.final_layer = nn.Sequential(
            upsampler(factor=4),
            nn.Sigmoid()
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
        nwarp_grid_3d = rearrange(nwarp_grid_3d, 'b c d h w -> b d c h w')

        return nwarp_grid_3d.contiguous()
    
    def sample_with_sort(self, cost, value):
        '''
        :param cost: [B, D, C, H, W]
        :param value: [B, D, H, W]
        :return: [B, K, C, H, W]
        '''

        b, d, c, h, w = cost.shape

        # topk method
        diff = torch.abs(self.costrange - value)
        _, indices = torch.topk(1.0 / (diff + 1e-6), k=self.option.model.dsample_num, dim=1)

        # sort indices
        sorted_indices = torch.sort(indices, dim=1)[0]
        squeezed_cost = torch.gather(cost, index=repeat(sorted_indices, 'b k h w -> b k c h w', c=c), dim=1)
        squeezed_disp = torch.gather(self.costrange.expand(b, d, h, w), index=sorted_indices, dim=1)

        return squeezed_cost, squeezed_disp
    
    def forward(self, costs, disp_maps, batch):
        '''
        :param cost: [B, C, D, H, W]
        :param disp: [B, 1, H, W]
        '''
        
        # params
        gt_depth = batch['depth'] if 'depth' in batch.keys() else None  # not used
        ab_value = batch['abvalue'] if 'abvalue' in batch.keys() else None
        
        # normal estimate
        normals, offset1s, offset2s = [], [], []
        for idx, cost in enumerate(costs):
            
            b, c, d, h, w = cost.shape
            cost = rearrange(cost, 'b c d h w -> b d c h w')
            disp = F.interpolate(disp_maps[idx].unsqueeze(1), scale_factor=0.25, mode='nearest') * 0.25
            
            # surface sampling
            if self.option.model.use_sampling:
                cost, disp_range = self.sample_with_sort(cost, disp)
            else:
                disp_range = self.costrange.expand(b, -1, h, w).type_as(cost)  # [B, D, H, W]
                
            # coordinate volume + sampled aggregated volume
            coordV = self.grid_maker_3d(cost, batch['K'], disp_range, ab_value)
            feature_volume = rearrange(torch.cat([cost, coordV], dim=2), 'b d c h w -> b c d h w')  # [B, C + 3, D, H, W]
            feature_volume = feature_volume.contiguous()
            
            # D3D module (3D deformable conv)
            if self.option.model.use_deform:
                '''
                feature_volume1,2 : [B, 2C, N, H, W]
                offset : [B, (3*3*3)*3, N, H, W]
                '''
                feature_volume1, offset1 = self.deform_conv1(feature_volume)
                feature_volume1 = self.act1(feature_volume1)

                feature_volume2, offset2 = self.deform_conv2(feature_volume1)
                feature_volume2 = self.act2(feature_volume2)
            else:
                feature_volume2 = self.original_conv(feature_volume)
                offset1, offset2 = None, None
                
            # normal conv applied (shared 2D conv)
            features = rearrange(feature_volume2, 'b c d h w -> (b d) c h w')
            last_feature = self.final_layer(self.n_convs(features))
            last_feature = rearrange(last_feature, '(b d) c h w -> b d c h w', b=b).mean(dim=1)
            
            # normalize from -1 to 1
            normals.append(last_feature * 2.0 - 1.0)
            offset1s.append(offset1)
            offset2s.append(offset2)
            
        return normals, offset1s, offset2s
            