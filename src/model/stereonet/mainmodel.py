from __future__ import print_function

import pdb
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
import torch.utils.data as torch_data
from einops import rearrange

from src.loss.loss_selector import loss_selector
from src.metric.metric_selector import metric_selector
from dataloader.loader_selector import loader_selector

from src.module.asm.basics import convbn_3d
from src.model.stereonet.modules import disp_regression, FeatureExtraction, EdgeAwareRefinement
from src.model.model_selector import optimizer_selector, scheduler_selector


'''

    Source from "https://github.com/meteorshowers/X-StereoLab"
    "Stereonet guided hierarchical refinement for real-time edge-aware depth prediction (ECCV 2018)"

'''


class STEREONET(pl.LightningModule):
    def __init__(self, option):
        super(STEREONET, self).__init__()
        self.save_hyperparameters()
        
        self.option = option
        self.mindisp = option.model.mindisp
        self.maxdisp = option.model.maxdisp
        self.level = math.pow(2, option.model.k)
        self.costrange = np.array(range(int(self.level))) * (
                (self.maxdisp / 4.0 - self.mindisp / 4.0) / float(self.level)) + self.mindisp / 4.0

        # use which backbone
        self.feature_extraction = FeatureExtraction(option.model.k, option.model.input_channel)
        
        self.filter = nn.ModuleList()
        for _ in range(4):
            self.filter.append(
                nn.Sequential(
                    convbn_3d(32, 32, kernel_size=3, stride=1, pad=1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        self.conv3d_alone = nn.Conv3d(
            32, 1, kernel_size=3, stride=1, padding=1)

        self.regression_layer = disp_regression(self.mindisp, self.maxdisp, self.level)
        
        self.edge_aware_refinements = nn.ModuleList()
        for _ in range(1):
            self.edge_aware_refinements.append(EdgeAwareRefinement(4))
            
        # loss and metric
        self.loss_model = loss_selector(option)
        self.metric_model = metric_selector(option)
            
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
                
    def forward(self, batch):
        
        # Feature Extraction : [Batch, channel, H / 4, W / 4]
        if 'groupname' in batch.keys() and not self.training:  # if the dataset is revised, will be removed
            if batch['groupname'][0] == '2020-2-9_group20':
                ref_fea = self.feature_extraction(batch['right'])
                target_fea = self.feature_extraction(batch['left'])
            else:
                ref_fea = self.feature_extraction(batch['left'])
                target_fea = self.feature_extraction(batch['right'])
        else:
            if self.option.dataset.flip_lr:
                ref_fea = self.feature_extraction(batch['right'])
                target_fea = self.feature_extraction(batch['left'])
            else:
                ref_fea = self.feature_extraction(batch['left'])
                target_fea = self.feature_extraction(batch['right'])
                
        # Cost Volume
        costv = torch.FloatTensor(ref_fea.size()[0],
                                  ref_fea.size()[1],
                                  int(self.level),
                                  ref_fea.size()[2],
                                  ref_fea.size()[3]).zero_().cuda()
        for i, disp in enumerate(self.costrange):
            disp = int(disp)

            if disp == 0:
                costv[:, :, i, :, :] = ref_fea - target_fea
            elif disp > 0:
                costv[:, :, i, :-disp, :] = ref_fea[:, :, :-disp, :] - target_fea[:, :, disp:, :]
            else:
                costv[:, :, i, -disp:, :] = ref_fea[:, :, -disp:, :] - target_fea[:, :, :disp, :]
        costv = costv.contiguous()
        
        # Aggregation
        for f in self.filter:
            costv = f(costv)
        costv = self.conv3d_alone(costv)
        costv = torch.squeeze(costv, 1)
        
        # Regression
        cost, cost_p = self.regression_layer([costv])
        
        pred_pyramid_list = [cost[0]]
        pred_pyramid_list.append(self.edge_aware_refinements[0](pred_pyramid_list[0], batch['right']))
        
        for i in range(1):
            pred_pyramid_list[i] = pred_pyramid_list[i]* (
                batch['right'].size()[-1] / pred_pyramid_list[i].size()[-1])
            pred_pyramid_list[i] = torch.squeeze(
            F.interpolate(
                torch.unsqueeze(pred_pyramid_list[i], dim=1),
                size=batch['right'].size()[-2:],
                mode='bilinear',
                align_corners=False), dim=1
            )
            
        # Results
        pred_pyramid = rearrange(pred_pyramid_list, 'n b h w -> b n h w')
        cost_p = rearrange(cost_p, 'n b c h w -> b n c h w')
        results = {'pred_depth': pred_pyramid, 
                   'prob_depth': cost_p,
                   'ref_feature': ref_fea.max(1)[0]}
        
        # Loss apply
        if self.training and 'disp' in batch.keys():
            loss = self.loss_model.forward(results, batch)
            results.update(loss)
            
        return results
    
    def train_dataloader(self):
        dataloader = torch_data.DataLoader(loader_selector(self.option, True), batch_size=self.option.batch_size,
                                           shuffle=True, num_workers=self.option.workers, drop_last=False,
                                           pin_memory=self.option.pin_memory)
        return dataloader
    
    def val_dataloader(self):
        dataloader = torch_data.DataLoader(loader_selector(self.option, False), batch_size=1,
                                           shuffle=False, num_workers=self.option.workers, drop_last=False,
                                           pin_memory=self.option.pin_memory)
        return dataloader
    
    def test_dataloader(self):
        dataloader = torch_data.DataLoader(loader_selector(self.option, False), batch_size=self.option.batch_size,
                                           shuffle=False, num_workers=self.option.workers, drop_last=False,
                                           pin_memory=self.option.pin_memory)
        return dataloader
    
    def training_step(self, batch, batch_idx):
        results = self.forward(batch)
        losses = dict()
        for key in results.keys():
            if key == 'final_loss':
                continue
            if 'loss' in key:
                losses.update({key: results[key]})
                self.log(key, results[key], prog_bar=True)

        return {'loss': results['final_loss'], 'log': losses}
    
    def validation_step(self, batch, batch_idx):
        # results = self.forward(batch)
        # if 'depth' in batch.keys():
        #     metrics = self.metric_model.forward(results, batch)
        # return results
        return None
    
    def validation_epoch_end(self, outputs):
        #self.metric_model.viewer()
        return None
    
    def test_step(self, batch, batch_idx):
        results = self.forward(batch)
        if 'depth' in batch.keys():
            metrics = self.metric_model.forward(results, batch)
        return results
    
    def test_epoch_end(self, outputs):
        if self.option.mode == 'test':
            self.metric_model.viewer()
        return None
    
    def configure_optimizers(self):
        optimizers, schedulers = [], []

        # optimizer selection
        optimizer = optimizer_selector(self.parameters(), self.option)
        optimizers.append(optimizer)

        # scheduler selection
        scheduler = scheduler_selector(optimizer, self.option)
        if scheduler is not None:
            schedulers.append(scheduler)

        return optimizers, schedulers

        
        
        