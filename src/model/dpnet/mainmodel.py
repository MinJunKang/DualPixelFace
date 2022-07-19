from __future__ import print_function

import pdb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
import torch.utils.data as torch_data
from einops import rearrange

from src.loss.loss_selector import loss_selector
from src.metric.metric_selector import metric_selector
from dataloader.loader_selector import loader_selector

from src.module.asm.basics import BasicBlock, depthwise_separable_conv
from src.model.model_selector import optimizer_selector, scheduler_selector
from src.model.dpnet.modules import Encoder, Encoder2, Decoder, Decoder2


'''

    Reimplemented version of "Learning Single Camera Depth Estimation using Dual-Pixels (ICCV 19)"

'''


class DPNET(pl.LightningModule):
    def __init__(self, option):
        super(DPNET, self).__init__()
        self.save_hyperparameters()
        
        self.option = option
        self.mindisp = option.model.mindisp
        self.maxdisp = option.model.maxdisp
        
        # mode : use bilinar upsampling instead of deconvolution
        mode = None
        
        # Encoder
        self.enc_layer1_1 = Encoder2(option.model.input_channel * 2, 8, 2)
        self.enc_layer1_2 = Encoder(8 + option.model.input_channel * 2, 11, 11, 1, 1)

        self.enc_layer2_1 = Encoder(11, 16, 32, 2, 0)
        self.enc_layer2_2 = Encoder(32, 16, 32, 1, 1)
        self.enc_layer2_3 = Encoder(32, 16, 32, 1, 1)

        self.enc_layer3_1 = Encoder(32, 16, 64, 2, 2)
        self.enc_layer3_2 = Encoder(64, 16, 64, 1, 1)
        self.enc_layer3_3 = Encoder(64, 16, 64, 1, 1)

        self.enc_layer4_1 = Encoder(64, 32, 128, 2, 1)
        self.enc_layer4_2 = Encoder(128, 32, 128, 1, 1)
        self.enc_layer4_3 = Encoder(128, 32, 128, 1, 1)

        self.enc_layer5_1 = Encoder(128, 32, 128, 2, 1)
        self.enc_layer5_2 = Encoder(128, 32, 128, 1, 1)
        self.enc_layer5_3 = Encoder(128, 32, 128, 1, 1)

        # Decoder
        self.dec_layer1 = Decoder(32, 16, 4, 1, 0, 1, mode=mode)
        self.dec_layer2 = Decoder(64, 16, 4, 0, 0, 0, mode=mode)
        self.dec_layer3 = Decoder(128, 16, 2, 0, 1, 0, mode=mode)
        self.dec_layer4 = Decoder(128, 32, 1, 1, 1, 1, mode=mode)
        
        # Skip Connection
        self.skip_layer1 = depthwise_separable_conv(11, 16, kernel_size=3, padding=3)
        self.skip_layer2 = depthwise_separable_conv(32, 16, kernel_size=3, padding=3)
        self.skip_layer3 = depthwise_separable_conv(64, 16, kernel_size=3, padding=3)
        self.skip_layer4 = depthwise_separable_conv(128, 32, kernel_size=3, padding=2)

        # Basic Blocks
        self.dec_layer1_b = BasicBlock(16, 32, kernel_size=1, pad=1, bn=False, relu=False)
        self.dec_layer2_b = BasicBlock(16, 32, kernel_size=1, pad=1, bn=False, relu=False)
        self.dec_layer3_b = BasicBlock(16, 64, kernel_size=1, pad=1, bn=False, relu=False)
        self.dec_layer4_b = BasicBlock(32, 128, kernel_size=1, pad=1, bn=False, relu=False)

        # Upsample layer (Last layer)
        self.last_layer = Decoder2(32, 8, 8, 4, 1, 0, 1, mode=mode)
        self.conv_last_layer5 = BasicBlock(128, 1, kernel_size=7, pad=1)
        self.conv_last_layer4 = BasicBlock(64, 1, kernel_size=7, pad=0)
        self.conv_last_layer3 = BasicBlock(32, 1, kernel_size=7, pad=1)
        self.conv_last_layer2 = BasicBlock(32, 1, kernel_size=7, pad=1)
        self.conv_last_layer1 = BasicBlock(8, 1, kernel_size=7, pad=1)
        
        # Activation function
        self.prelu = nn.PReLU(init=0.05)
        
        # loss and metric
        self.loss_model = loss_selector(option)
        self.metric_model = metric_selector(option)
        
        # Weight initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        self.init_weights()
                
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, batch):
        
        # Inputs
        if 'groupname' in batch.keys() and not self.training:  # if the dataset is revised, will be removed
            if batch['groupname'][0] == '2020-2-9_group20':
                ref_img, target_img = batch['right'], batch['left']
            else:
                ref_img, target_img = batch['left'], batch['right']
        else:
            if self.option.dataset.flip_lr:
                ref_img, target_img = batch['right'], batch['left']
            else:
                ref_img, target_img = batch['left'], batch['right']
                
        # Concatenate
        x = torch.cat((ref_img, target_img), dim=1)

        # Encoder layer 1
        x_layer1 = self.enc_layer1_1(x)
        x_layer1 = self.enc_layer1_2(x_layer1)

        # Encoder layer 2
        x_layer2 = self.enc_layer2_1(x_layer1)
        x_layer2 = self.enc_layer2_2(x_layer2)
        x_layer2 = self.enc_layer2_3(x_layer2)

        # Encoder layer 3
        x_layer3 = self.enc_layer3_1(x_layer2)
        x_layer3 = self.enc_layer3_2(x_layer3)
        x_layer3 = self.enc_layer3_3(x_layer3)

        # Encoder layer 4
        x_layer4 = self.enc_layer4_1(x_layer3)
        x_layer4 = self.enc_layer4_2(x_layer4)
        x_layer4 = self.enc_layer4_3(x_layer4)

        # Encoder layer 5
        x_layer5 = self.enc_layer5_1(x_layer4)
        x_layer5 = self.enc_layer5_2(x_layer5)
        x_layer5 = self.enc_layer5_3(x_layer5)

        # Decoder layer 5
        y_layer5 = self.dec_layer4(x_layer5)
        y_layer5 = self.prelu(y_layer5 + self.skip_layer4(x_layer4))  # torch.Size([1, 32, 50, 34])
        y_layer5 = self.dec_layer4_b(y_layer5)  # torch.Size([1, 128, 52, 36])

        # Decoder layer 4
        y_layer4 = self.dec_layer3(y_layer5)
        y_layer4 = self.prelu(y_layer4 + self.skip_layer3(x_layer3))  # torch.Size([1, 16, 100, 68])
        y_layer4 = self.dec_layer3_b(y_layer4)

        # Decoder layer 3
        y_layer3 = self.dec_layer2(y_layer4)
        y_layer3 = self.prelu(y_layer3 + self.skip_layer2(x_layer2))  # torch.Size([1, 16, 194, 130])
        y_layer3 = self.dec_layer2_b(y_layer3)

        # Decoder layer 2
        y_layer2 = self.dec_layer1(y_layer3)
        y_layer2 = self.prelu(y_layer2 + self.skip_layer1(x_layer1))  # torch.Size([1, 16, 386, 258])
        y_layer2 = self.dec_layer1_b(y_layer2)

        # Decoder layer 1
        y_layer1 = self.last_layer(y_layer2)
        
        # outputs layer
        out5 = torch.squeeze(F.interpolate(self.conv_last_layer5(y_layer5), scale_factor=16,
                                           mode='bilinear', align_corners=True), dim=1)
        out4 = torch.squeeze(F.interpolate(self.conv_last_layer4(y_layer4), scale_factor=8,
                                           mode='bilinear', align_corners=True), dim=1)
        out3 = torch.squeeze(F.interpolate(self.conv_last_layer3(y_layer3), scale_factor=4,
                                           mode='bilinear', align_corners=True), dim=1)
        out2 = torch.squeeze(F.interpolate(self.conv_last_layer2(y_layer2), scale_factor=2,
                                           mode='bilinear', align_corners=True), dim=1)
        out1 = torch.squeeze(self.conv_last_layer1(y_layer1), dim=1)
        
        pred_final = rearrange([out1, out2, out3, out4, out5], 'n b h w -> b n h w')
        
        results = {'pred_depth': pred_final, 
                   'ref_feature': x_layer1.max(1)[0]}
        
        # Loss apply
        if self.training and 'depth' in batch.keys():
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
        # self.metric_model.viewer()
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