from __future__ import print_function

import pdb
from einops import rearrange
import pytorch_lightning as pl
import torch.utils.data as torch_data

from src.loss.loss_selector import loss_selector
from src.metric.metric_selector import metric_selector
from dataloader.loader_selector import loader_selector

from src.model.bts.modules import Encoder, Decoder
from src.model.model_selector import optimizer_selector, scheduler_selector


'''

    Source from "https://github.com/cleinc/bts"
    of "From big to small: Multi-scale local planar guidance for monocular depth estimation (arxiv)"

'''


class BTS(pl.LightningModule):
    
    def __init__(self, option):
        super(BTS, self).__init__()
        self.save_hyperparameters()
        
        self.option = option
        
        # loss and metric
        self.loss_model = loss_selector(option)
        self.metric_model = metric_selector(option)
        
        # Encoder and Decoder
        self.encoder = Encoder(option)
        self.decoder = Decoder(option, self.encoder.feat_out_channels, option.model.bts_size)
        
    def forward(self, batch):
        
        # Encoder - Decoder
        skip_feat = self.encoder(batch['center'])
        out = self.decoder(skip_feat)
        
        # Results
        pred_final = rearrange(list(out), 'n b 1 h w -> b n h w')
        results = {'pred_depth': pred_final[:, 0:1], 
                   'ref_feature': skip_feat[0].max(1)[0]}
        
        # Loss apply
        if self.training and 'depth' in batch.keys():
            loss = self.loss_model.forward(results, batch, target_type='depth')
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
        # pdb.set_trace()
        # if 'depth' in batch.keys():
        #     metrics = self.metric_model.forward(results, batch, target_type='depth')
        # return results
        return None
    
    def validation_epoch_end(self, outputs):
        # self.metric_model.viewer()
        return None
    
    def test_step(self, batch, batch_idx):
        results = self.forward(batch)
        if 'depth' in batch.keys():
            metrics = self.metric_model.forward(results, batch, target_type='depth')
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