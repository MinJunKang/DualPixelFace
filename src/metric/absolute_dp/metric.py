
import numpy as np


def compute_errors_test_depth(gt, pred, mask, threshold):
    '''
    :param gt: ground truth depth [B, H, W]
    :param pred: predicted depth [B, H, W]
    :param mask: mask (given) [B, H, W]
    :param threshold: for a1, a2, a3
    :return:
    '''
    gt = gt[mask > 0]
    pred = pred[mask > 0]
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < threshold).mean()
    a2 = (thresh < threshold ** 2).mean()
    a3 = (thresh < threshold ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_diff = np.mean(np.abs(gt - pred))
    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return [abs_rel, abs_diff, sq_rel, rmse, rmse_log, a1, a2, a3]


