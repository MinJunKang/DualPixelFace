
import torch
import math
import torch.nn.functional as F


def calNormalAcc(gt_n, pred_n, mask=None):
    '''
    :param gt_n: [B, 3, H, W]
    :param pred_n: [B, 3, H, W]
    :param mask: mask (given) [B, 1, H, W]
    :return:
    '''

    gt_n = F.normalize(gt_n, dim=1)
    pred_n = F.normalize(pred_n, dim=1)

    dot_product = (gt_n * pred_n).sum(1).clamp(-1.0, 1.0)
    error_map = torch.acos(dot_product)  # [-pi, pi]
    angular_map = error_map * 180.0 / math.pi
    angular_map = angular_map * mask.narrow(1, 0, 1).squeeze(1)

    valid = mask.narrow(1, 0, 1).sum()
    ang_valid = angular_map[mask.narrow(1, 0, 1).squeeze(1).byte()]

    n_err_mean = ang_valid.sum() / valid

    return n_err_mean


def calNormalAccRMSE(gt_n, pred_n, mask=None):
    '''
    :param gt_n: [B, 3, H, W]
    :param pred_n: [B, 3, H, W]
    :param mask: mask (given) [B, 1, H, W]
    :return:
    '''

    gt_n = F.normalize(gt_n, dim=1)
    pred_n = F.normalize(pred_n, dim=1)

    dot_product = (gt_n * pred_n).sum(1).clamp(-1, 1)
    angular_map = torch.acos(dot_product)  # [-pi, pi]
    angular_map = angular_map * mask.narrow(1, 0, 1).squeeze(1)

    valid = mask.narrow(1, 0, 1).sum()
    ang_valid = angular_map[mask.narrow(1, 0, 1).squeeze(1).byte()]
    n_err_rmse = torch.sqrt((ang_valid ** 2).sum() / valid) * 180.0 / math.pi

    return n_err_rmse
