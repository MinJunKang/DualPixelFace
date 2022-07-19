from __future__ import print_function

import pdb
import math
import torch
import torch.nn as nn

import pytorch_lightning as pl
import torch.utils.data as torch_data
from einops import rearrange

from src.loss.loss_selector import loss_selector
from src.metric.metric_selector import metric_selector
from dataloader.loader_selector import loader_selector

from src.model.model_selector import optimizer_selector, scheduler_selector
from src.model.psmnet.modules import feature_extraction, CostVolume, PSMNetHGAggregation, disp_regression


'''

    Source from "https://github.com/JiaRenChang/PSMNet"
    of "Pyramid Stereo Matching Network (CVPR 2018)"
    
    (also included group-wise correlation module of "https://github.com/xy-guo/GwcNet")
    of "Group-wise Correlation Stereo Network (CVPR 2019)"

'''


class PSMNET(pl.LightningModule):
    def __init__(self, option):
        super(PSMNET, self).__init__()
        self.save_hyperparameters()
        
        self.option = option
        self.mindisp = option.model.mindisp
        self.maxdisp = option.model.maxdisp
        self.level = option.model.level
        
        # Feature Extractor
        self.feature_extraction = feature_extraction(option)
        
        # Cost Volume
        self.cost_volume = CostVolume(option, self.mindisp, self.maxdisp)
        
        # Cost Aggregation
        self.aggregation = PSMNetHGAggregation(option)
        
        # defocus-disparity regressor
        self.regression_layer = disp_regression(self.mindisp, self.maxdisp, self.level)
        
        # loss and metric
        self.loss_model = loss_selector(option)
        self.metric_model = metric_selector(option)
        
        # initialize weights
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
        cost = self.cost_volume(ref_fea, target_fea)
        
        # Aggregation
        cost_i, cost = self.aggregation(cost)
        
        # Regression
        cost_f, cost_p = self.regression_layer(cost_i)
        
        # Results
        cost_f = rearrange(cost_f, 'n b h w -> b n h w')
        cost_p = rearrange(cost_p, 'n b c h w -> b n c h w')
        results = {'pred_depth': cost_f, 
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
        results = self.forward(batch)
        
        # save_result_fig_depth(self.option, results, batch, self.current_epoch, self.global_step, mode='TEST')
        if 'depth' in batch.keys():
            metrics = self.metric_model.forward(results, batch)
        return results
    
    def validation_epoch_end(self, outputs):
        self.metric_model.viewer()
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
        
        
        
        
        
        