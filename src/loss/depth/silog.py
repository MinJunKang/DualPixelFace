
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.geometry import regress_affine, inverse_depth, depth2disp


class SILOGLoss(nn.modules.Module):
    def __init__(self, option):
        super(SILOGLoss, self).__init__()
        self.target_type = option.model.target_type
        self.variance_focus = option.model.variance_focus
        self.conversion_method = option.dataset.dp_conversion  # given or least_square
        self.weights = torch.tensor(option.model.loss_weight)
        
    def forward(self, preds, batch, target_type='disp'):
        '''
            consider the case of disparity target or depth target or idepth target
        '''
        
        pred = preds['pred_depth']
        num_pred = pred.shape[1]
        weights = torch.tensor([1.0]).type_as(pred) if num_pred == 1 else self.weights.type_as(pred)
        assert (num_pred == len(weights))
        assert (target_type in ['disp', 'depth', 'idepth'])
        
        mask = batch['mask'] > 0 if 'mask' in batch.keys() else None
        pred_ = pred if target_type in ['disp', 'idepth'] else inverse_depth(pred)
        if self.conversion_method == 'least_square' or not 'abvalue' in batch.keys():
            ab_value = regress_affine(pred[:, 0:1], batch['idepth'].unsqueeze(1))
            gt = depth2disp(batch['depth'].unsqueeze(1), ab_value)
        else:
            ab_value = batch['abvalue']
            gt = batch['disp'] if target_type == 'disp' else batch['idepth']
        
        if 'conf' in batch.keys():
            if batch['conf'] is not None:
                pred_ = pred_ * batch['conf'].unsqueeze(1)
                gt = gt * batch['conf']
                
        if mask is not None:
            dist = [weights[i] * (torch.log(pred_[mask]) - torch.log(gt[mask])) for i in range(num_pred)]
        else:
            dist = [weights[i] * (torch.log(pred_) - torch.log(gt)) for i in range(num_pred)]
        loss = sum([torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0 for d in dist])

        results = {'loss': loss, 'abvalue': ab_value}

        return results
            
            
            
        
        