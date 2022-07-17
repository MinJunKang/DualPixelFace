from __future__ import print_function

import pdb
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import pytorch_lightning as pl
import torch.utils.data as torch_data
from einops import rearrange

from src.loss.loss_selector import loss_selector
from src.metric.metric_selector import metric_selector
from dataloader.loader_selector import loader_selector

from src.module.asm.basics import convbn_3d
from src.model.model_selector import optimizer_selector, scheduler_selector
from src.model.nnet.modules import convtext, feature_extraction, CostVolume, disp_regression
from src.model.nnet.normal_module_ import NormalModule


'''

    Source from "https://github.com/udaykusupati/Normal-Assisted-Stereo"
    "Normal Assisted Stereo Depth Estimation (CVPR 2020)"

'''


class NNET(pl.LightningModule):
    def __init__(self, option):
        super(NNET, self).__init__()
        self.save_hyperparameters()
        
        self.option = option
        self.mindisp = option.model.mindisp
        self.maxdisp = option.model.maxdisp
        self.level = option.model.level
        inplanes = option.model.inplanes
        
        # backbone
        self.feature_extraction = feature_extraction(option)
        
        # cost volume
        self.cost_volume = CostVolume(option, self.mindisp, self.maxdisp)
        
        # MVS Style aggregation
        self.convs = nn.Sequential(
            convtext(inplanes + 1, inplanes * 4, 3, 1, 1),
            convtext(inplanes * 4, inplanes * 4, 3, 1, 2),
            convtext(inplanes * 4, inplanes * 4, 3, 1, 4),
            convtext(inplanes * 4, inplanes * 3, 3, 1, 8),
            convtext(inplanes * 3, inplanes * 2, 3, 1, 16),
            convtext(inplanes * 2, inplanes, 3, 1, 1),
            convtext(inplanes, 1, 3, 1, 1)
        )
        
        # aggregation
        self.dres0 = nn.Sequential(convbn_3d(inplanes * 2, inplanes, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(inplanes, inplanes, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(inplanes, inplanes, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(inplanes, inplanes, 3, 1, 1)) 

        self.dres2 = nn.Sequential(convbn_3d(inplanes, inplanes, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(inplanes, inplanes, 3, 1, 1))
 
        self.dres3 = nn.Sequential(convbn_3d(inplanes, inplanes, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(inplanes, inplanes, 3, 1, 1)) 

        self.dres4 = nn.Sequential(convbn_3d(inplanes, inplanes, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(inplanes, inplanes, 3, 1, 1)) 
 
        self.classify = nn.Sequential(convbn_3d(inplanes, inplanes, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(inplanes, 1, kernel_size=3, padding=1, stride=1,bias=False))
        
        # disparity regression
        self.regression_layer = disp_regression(self.mindisp, self.maxdisp, self.level)
        
        # normal module
        self.normal_module = NormalModule(option, self.mindisp, self.maxdisp) if option.model.predict_normal else None
        
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
        cost = self.cost_volume(ref_fea, target_fea)
        
        # Aggregation
        cost0 = self.dres0(cost)
        cost_in0 = cost0.clone()
        cost0 = self.dres1(cost0) + cost0
        cost0 = self.dres2(cost0) + cost0 
        cost0 = self.dres3(cost0) + cost0 
        cost0 = self.dres4(cost0) + cost0
        cost_in = torch.cat((cost_in0, cost0.clone()), dim = 1)
        costs = self.classify(cost0)
        
        costss = Variable(torch.FloatTensor(ref_fea.size()[0], 1, self.level,  ref_fea.size()[2],  ref_fea.size()[3]).zero_()).cuda()
        for i in range(self.level):
            costt = costs[:, :, i, :, :]
            costss[:, :, i, :, :] = self.convs(torch.cat([ref_fea, costt],1)) + costt
            
        # disparity regression
        costs = F.interpolate(costs, scale_factor=4, mode='trilinear', align_corners=False)
        costs = torch.squeeze(costs, 1)
        costss = F.interpolate(costss, scale_factor=4, mode='trilinear', align_corners=False)
        costss = torch.squeeze(costss,1)
        costf, costf_p = self.regression_layer([costs, costss])
        
        # Normal branch
        normal_results = self.normal_module(cost_in, batch) if self.option.model.predict_normal else None
        
        # Results
        cost_f = rearrange(costf, 'n b h w -> b n h w')
        cost_p = rearrange(costf_p, 'n b c h w -> b n c h w')
        normal = rearrange(normal_results, 'n b c h w -> b n c h w')
        results = {'pred_depth': cost_f, 
                   'prob_depth': cost_p, 
                   'pred_normal': normal,
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
        save_result_fig_depth(self.option, results, batch, self.current_epoch, self.global_step, mode='TEST')
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
        
        
        