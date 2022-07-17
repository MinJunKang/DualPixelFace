
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.geometry import regress_affine, inverse_depth, depth2disp


class SMOOTHL1Loss(nn.modules.Module):
    def __init__(self, option):
        super(SMOOTHL1Loss, self).__init__()
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
            loss = sum([weights[i] * F.smooth_l1_loss(pred_[:, i][mask], gt[mask],
                                                      size_average=True) for i in range(num_pred)])
        else:
            loss = sum([weights[i] * F.smooth_l1_loss(pred_[:, i], gt,
                                                      size_average=True) for i in range(num_pred)])

        results = {'loss': loss, 'abvalue': ab_value}

        return results
            
            
            
        
        