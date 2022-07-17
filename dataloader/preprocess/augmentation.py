from __future__ import division

import pdb
import sys
import cv2
import random
import torch
import numpy as np
from PIL import Image

import torchvision.transforms.functional as F

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


# Compose transforms
class Compose(object):
    """ Composes several co_transforms together.
    """
    def __init__(self, co_transforms):
        self.co_transforms = co_transforms
        
    def __call__(self, inputs, targets):
        for t in self.co_transforms:
            inputs, targets = t(inputs, targets)
        return inputs, targets
    
    
# Numpy transform
class ToNumpy(object):
    """ Transform to numpy instance
    """
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, inputs, targets):

        for idx, input in enumerate(inputs):
            if input is not None:
                input = np.asarray(input)
                if self.dtype is not None:
                    input = input.astype(self.dtype)
                inputs[idx] = input
        return inputs, targets
    
    
# PIL transform
class ToPIL(object):
    """ Transform to PIL instance
    """
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, inputs, targets):
        for idx, input in enumerate(inputs):
            if input is not None:
                if self.dtype is not None:
                    input = input.astype(self.dtype)
                inputs[idx] = Image.fromarray(input)
        return inputs, targets
    
    
# To tensor
class ToTensor(object):
    """ Transform to tensor instance
    """
    def __init__(self, squeeze=True):
        self.squeeze = squeeze

    def __call__(self, inputs, targets):

        for idx, input in enumerate(inputs):
            if input is not None:
                if self.squeeze:
                    inputs[idx] = F.to_tensor(input).squeeze()
                else:
                    inputs[idx] = F.to_tensor(input)

        for idx, target in enumerate(targets):
            if target is not None:
                if self.squeeze:
                    targets[idx] = F.to_tensor(target).squeeze()
                else:
                    targets[idx] = F.to_tensor(target)
        return inputs, targets
    
    
# Cropper
class Cropper(object):

    def __init__(self, option, cropped_size):
        self.min_inlier = option.min_inlier
        self.max_trial = option.max_trial
        self.method = option.method
        self.cropped_size = cropped_size
        self.coords = [0, 0]

    def ROIselector(self, target):

        coords = np.argwhere(target > 0)
        roiy = int(np.min(coords[:, 0]))
        roix = int(np.min(coords[:, 1]))

        return roix, roiy

    def imgselector(self, inputs):
        if inputs is None:
            raise NotImplementedError('inputs must not be None')
        img = None
        for input in inputs:
            if input is not None:
                img = input
                break

        if img is None:
            raise NotImplementedError('size must not be None')
        return img

    def center_crop(self, inputs):
        img = self.imgselector(inputs)
        h = img.shape[0]
        w = img.shape[1]
        th = self.cropped_size[0]
        tw = self.cropped_size[1]
        j = int(round((h - th) / 2.))
        i = int(round((w - tw) / 2.))

        return i, j, tw, th

    def random_crop(self, inputs):
        img = self.imgselector(inputs)
        h = img.shape[0]
        w = img.shape[1]
        th = self.cropped_size[0]
        tw = self.cropped_size[1]
        j = random.randint(0, h - th)
        i = random.randint(0, w - tw)

        return i, j, tw, th

    def mask_random_crop(self, inputs, mask, roix, roiy):
        img = self.imgselector(inputs)
        h = img.shape[0]
        w = img.shape[1]
        th = self.cropped_size[0]
        tw = self.cropped_size[1]

        # inner loop
        cnt = 0
        while True:
            j = random.randint(roiy, h - th)
            i = random.randint(roix, w - tw)
            ratio = np.sum(mask[j: j + th, i: i + tw]) / (th * tw)
            if ratio >= self.min_inlier:
                break
            cnt += 1
            if cnt >= self.max_trial:
                j = random.randint(0, h - th)
                i = random.randint(0, w - tw)
                break

        return i, j, tw, th

    def applier(self, img, i, j, tw, th):

        if img.ndim == 3:
            return img[j:j + th, i:i + tw, :]
        elif img.ndim == 2:
            return img[j:j + th, i:i + tw]
        else:
            raise RuntimeError('img should be ndarray with 2 or 3 dimensions. Got {}'.format(img.ndim))

    def __call__(self, inputs, targets):

        if self.method == 'center_crop':  # center crop

            i, j, tw, th = self.center_crop(inputs)

        elif self.method == 'random_crop':  # random crop

            i, j, tw, th = self.random_crop(inputs)

        elif self.method == 'mask_random_crop':  # mask aware random crop
            if targets[1] is not None:
                roix, roiy = self.ROIselector(targets[1])
                i, j, tw, th = self.mask_random_crop(inputs, targets[1], roix, roiy)
            else:
                i, j, tw, th = self.random_crop(inputs)
        else:
            raise NotImplementedError('invalid cropping method')

        for n, input in enumerate(inputs):
            if input is not None:
                inputs[n] = self.applier(input, i, j, tw, th)
        for n, target in enumerate(targets):
            if target is not None:
                targets[n] = self.applier(target, i, j, tw, th)

        # save starting coordinate points
        self.coords[0] = i
        self.coords[1] = j
        
        return inputs, targets
    
    
class PhotometricAug(object):
    """ Photometric augmentation : brightness, gamma, contrast
    """

    def __init__(self, brightness, gamma, contrast):
        self.brightness = brightness
        self.gamma = gamma
        self.contrast = contrast

    def __call__(self, inputs, targets):

        for n, input in enumerate(inputs):
            if input is not None:
                if self.brightness != 0:
                    input = F.adjust_brightness(input, self.brightness)
                if self.gamma != 0:
                    input = F.adjust_gamma(input, self.gamma)
                if self.contrast != 0:
                    input = F.adjust_contrast(input, self.contrast)
            inputs[n] = input
        return inputs, targets


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd):
        self.alphastd = alphastd
        self.eigval = torch.tensor([0.2175, 0.0188, 0.0045])
        self.eigvec = torch.tensor([[-0.5675, 0.7192, 0.4009],
                                    [-0.5808, -0.0045, -0.8140],
                                    [-0.5836, -0.6948, 0.4203]])

    def __call__(self, inputs, targets):
        if self.alphastd == 0:
            return inputs, targets

        for i, img in enumerate(inputs):
            if img is None:
                continue
            if img.ndim == 3:
                alpha = img.new().resize_(3).normal_(0, self.alphastd)
                rgb = self.eigvec.type_as(img).clone()\
                    .mul(alpha.view(1, 3).expand(3, 3))\
                    .mul(self.eigval.view(1, 3).expand(3, 3))\
                    .sum(1).squeeze()
                inputs[i] = img.add(rgb.view(3, 1, 1).expand_as(img))
        return inputs, targets
    
    
# normalize tensor
class Normalizer(object):
    """Normalize a tensor image with mean and standard deviation.
        Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
        will normalize each channel of the input ``torch.*Tensor`` i.e.
        ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, inplace=False):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.inplace = inplace

    def __call__(self, inputs, targets):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """

        for i, tensor in enumerate(inputs):
            if tensor is None:
                continue
            tensor = tensor.float()
            if tensor.ndim == 2:
                mean = [0.5, ]
                std = [0.5, ]
                tensor = tensor.unsqueeze(0)
            else:
                mean = self.mean
                std = self.std
            if (2, 7) <= sys.version_info < (3, 1):
                inputs[i] = F.normalize(tensor, mean, std)
            else:
                inputs[i] = F.normalize(tensor, mean, std, self.inplace)
        return inputs, targets