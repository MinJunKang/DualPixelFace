
import pdb
import torch
import numpy as np
from .file_manager import tensor2numpy, error_handler
from scipy.optimize import lsq_linear, least_squares

'''
Functions for DUAL PIXEL 3D Geometry

abvalue = [a, b]
a = (-1) * L * g / (1 - f/g)
b = L / (1 - f / g) + tx * fx / 2
disparity = a / depth + b
depth = a / (disparity - b)

'''


# disparity to depth converter, torch / numpy version
def disp2depth(pred, abvalue, as_numpy=False):

    '''
    suppose disp = A / depth + B  (disparity -> depth)
    :param pred: [B, num_pred, H, W], disparity (Tensor)
    :param abvalue: [B, 2] (Tensor)
    :return depth : [B, num_pred, H, W]
    :param as_numpy: final result convert to numpy
    '''
    error_handler(isinstance(pred, torch.Tensor), 'input must be tensor', __name__, True)
    error_handler(isinstance(abvalue, torch.Tensor), 'input must be tensor', __name__, True)
    error_handler(len(pred.shape) == 4, 'shape is not matched', __name__, True)
    error_handler(len(abvalue.shape) == 2, 'shape is not matched', __name__, True)

    a = abvalue[:, 1].view(pred.shape[0], 1, 1, 1).contiguous().type_as(pred)
    b = abvalue[:, 0].view(pred.shape[0], 1, 1, 1).contiguous().type_as(pred)
    divider = (pred - b)
    depth = torch.div(a, divider)
    mask = torch.isnan(depth) | torch.isinf(depth)
    depth[mask] = 0.0

    if as_numpy:
        return tensor2numpy(depth)
    else:
        return depth


# depth to disparity converter, torch / numpy version
def depth2disp(pred, abvalue, as_numpy=False):

    '''
    suppose disp = A / depth + B  (depth -> disparity)
    :param pred: [B, num_pred, H, W], depth (Tensor)
    :param abvalue: [B, 2] (Tensor)
    :return disp : [B, num_pred, H, W], disparity
    :param as_numpy: final result convert to numpy
    '''

    error_handler(isinstance(pred, torch.Tensor), 'input must be tensor', __name__, True)
    error_handler(isinstance(abvalue, torch.Tensor), 'input must be tensor', __name__, True)
    error_handler(len(pred.shape) == 4, 'shape is not matched', __name__, True)
    error_handler(len(abvalue.shape) == 2, 'shape is not matched', __name__, True)

    a = abvalue[:, 1].view(pred.shape[0], 1, 1, 1).contiguous().type_as(pred)
    b = abvalue[:, 0].view(pred.shape[0], 1, 1, 1).contiguous().type_as(pred)

    disparity = torch.div(a, pred) + b
    mask = torch.isnan(disparity) | torch.isinf(disparity)
    disparity[mask] = -100.0

    if as_numpy:
        return tensor2numpy(disparity)
    else:
        return disparity
    
    
# regress a, b from ground truth
def regress_affine(pred, gt, as_numpy=False):

    '''
    d = A * 1 / Z* + B (conversion to disparity is possible)
    1 / Z * A + B = 1 / Z* (conversion to inverse depth is possible)
    :param pred: [B, 1, H, W], predicted disparity or inverse depth (Tensor)
    :param gt: [B, 1, H, W], inverse gt depth (Tensor)
    :param as_numpy: final result convert to numpy
    :return:
    '''

    error_handler(isinstance(pred, torch.Tensor), 'pred must be tensor', __name__, True)
    error_handler(isinstance(gt, torch.Tensor), 'gt must be tensor', __name__, True)
    error_handler(len(pred.shape) == 4, 'shape is not matched', __name__, True)
    error_handler(len(gt.shape) == 4, 'shape is not matched', __name__, True)

    # least square solution
    with torch.no_grad():
        abvalue = np.zeros((pred.shape[0], 2))
        for i in range(pred.shape[0]):
            target_ = tensor2numpy(pred[i].flatten())
            source_ = tensor2numpy(gt[i].flatten())
            mask = source_ > 0
            res = lsq_linear(np.stack([source_[mask], np.ones_like(source_[mask])], -1), target_[mask])
            res_lsq = least_squares((lambda x, A, b: A * x[0] + x[1] - b), res.x, loss='soft_l1', 
                                    f_scale=0.1, args=(source_[mask], target_[mask]))
            abvalue[i, :] = res_lsq.x[[1, 0]]
        
    '''
    avalue = X[:, 1]  # (B, 1) : A
    bvalue = X[:, 0]  # (B, 1) : B
    '''

    if as_numpy:
        return abvalue
    else:
        return torch.from_numpy(abvalue).type_as(pred)
    
    
# inversion depth : convert to invert depth
def inverse_depth(pred, as_numpy=False):

    '''
    :param pred: [B, num_pred, H, W], depth (Tensor)
    :return: [B, num_pred, H, W], inverse depth
    :param as_numpy: final result convert to numpy
    '''

    error_handler(isinstance(pred, torch.Tensor), 'input must be tensor', __name__, True)
    error_handler(len(pred.shape) == 4, 'shape is not matched', __name__, True)

    inverse_depth = torch.div(1.0, pred)
    mask = torch.isnan(inverse_depth) | torch.isinf(inverse_depth)
    inverse_depth[mask] = 0.0

    if as_numpy:
        return tensor2numpy(inverse_depth)
    else:
        return inverse_depth
    
    
'''
Functions for intrinsic, extrinsic conversion
'''


def intrinsic2KD(intrinsic):
    '''
    :param intrinsic: [9,] arrays
    :return: K [3, 3] intrinsics, D [4,] distortion factor
    '''

    K = np.zeros((3, 3))
    D = np.zeros(4)

    # Compose intrinsic matrix
    K[0][0] = intrinsic[0]
    K[0][1] = intrinsic[2]
    K[0][2] = intrinsic[3]
    K[1][1] = intrinsic[1]
    K[1][2] = intrinsic[4]
    K[2][2] = 1

    # Compose distortion params
    D[0] = intrinsic[5]
    D[1] = intrinsic[6]
    D[2] = intrinsic[7]
    D[3] = intrinsic[8]

    return K, D