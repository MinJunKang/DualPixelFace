
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from src.utils.geometry import regress_affine, inverse_depth


class COSINELoss(nn.modules.Module):
    def __init__(self, option):
        super(COSINELoss, self).__init__()
        self.weights = torch.tensor(option.model.loss_weight)
        
    def cosine_embedding_loss(self, input1, input2, eps=1e-6):
        '''
            To prevent overflow with fp16, we implement cosine embedding loss function by ourselves
        '''

        # compute cosine similarity
        denorm1 = torch.norm(input1, p=2, dim=-1, keepdim=True)
        denorm2 = torch.norm(input2, p=2, dim=-1, keepdim=True)
        denorm = (denorm1 * denorm2).clamp_min(eps)  # in case for fp16
        sim = ((input1 * input2) / denorm).clamp(min=-1.0, max=1.0)

        return torch.mean(1.0 - sim)  # 1 - cos(Npred, Ngt)
    
    def smoothl1loss(self, input1, input2):
        return F.smooth_l1_loss(input1, input2)

    def normalize(self, input, p=2, dim=-1, eps=1e-6):
        denorm = torch.norm(input, p=p, dim=dim, keepdim=True).clamp_min(eps)  # in case for fp16
        return input / denorm

    def forward(self, preds, batch, target_type=None):

        pred = preds['pred_normal']
        num_pred = pred.shape[1]
        weights = torch.tensor([1.0]).type_as(pred) if num_pred == 1 else self.weights.type_as(pred)
        assert(num_pred == len(weights))
        
        # (normal range from -1 to 1)
        mask = batch['mask'] > 0 if 'mask' in batch.keys() else None
        if mask is not None:
            pred = self.normalize(rearrange(pred, 'b n c h w -> b h w n c')[mask], p=2, dim=-1, eps=1e-6)
            gt = self.normalize(rearrange(batch['normal'], 'b c h w -> b h w c')[mask], p=2, dim=-1, eps=1e-6)
        else:
            pred = self.normalize(pred, p=2, dim=2, eps=1e-6)
            gt = self.normalize(batch['normal'], p=1, dim=2, eps=1e-6)
            
        loss = sum([weights[i] * self.cosine_embedding_loss(pred[:, i], gt) for i in range(num_pred)])
        
        results = {'loss': loss}

        return results