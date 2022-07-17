
import pdb
import math
import torch
import numpy as np

from src.utils.file_manager import error_handler
import dataloader.preprocess.augmentation as transforms


def is_square_and_even(integer):
    root = math.sqrt(integer)
    return (integer == int(root + 0.5) ** 2) and (int(root + 0.5) % 2 == 0)


class basic_transform(object):
    
    def __init__(self, option):
        self.opt = option
        self.coords = [0, 0]

    def check_inf_nan(self, tensors):
        for tensor in tensors:
            if tensor is not None:
                if (torch.sum(torch.isinf(tensor)) > 0) or (torch.sum(torch.isnan(tensor)) > 0):
                    return False
        return True

    def get_size(self, inputs, ratio, factor):

        if inputs is None:
            raise NotImplementedError('inputs must not be None')
        size = None
        for input in inputs:
            if input is not None:
                size = (input.shape[0], input.shape[1])
                break

        if size is None:
            raise NotImplementedError('size must not be None')

        n = np.ceil(np.array(size) * ratio // factor).astype('int')

        return tuple(factor * n)

    def apply(self, inputs_, targets_):
        '''
        :param inputs_: list of tensors (numpy array), check if the data is None or not, [H, W, C]
        :param targets_: list of tensors (numpy array), check if the data is None or not, [H, W]
        :return: augmented inputs_, augmented targets_ : inputs, targets
        '''

        # transform all the inputs/targets to numpy
        inputs, targets = transforms.ToNumpy()(inputs_.copy(), targets_.copy())
        
        # apply augmentation (crop augmentation, photometric augmentation)
        if 'crop_aug' in self.opt.augmentation:
            crop_aug_opt = self.opt.crop_aug
            if crop_aug_opt.type == 'soft_crop':
                cropped_size = self.get_size(inputs, crop_aug_opt.soft_crop.crop_ratio, crop_aug_opt.soft_crop.crop_factor)
            else:
                cropped_size = (crop_aug_opt.hard_crop.crop_height, crop_aug_opt.hard_crop.crop_width)
            crop_aug = transforms.Cropper(crop_aug_opt, cropped_size)
            inputs, targets = crop_aug(inputs, targets)
            self.coords = crop_aug.coords
            
        if 'photo_aug' in self.opt.augmentation:
            photo_aug_opt = self.opt.photo_aug
            random_brightness = np.random.uniform(0.7, 1.2, 1)[0] if photo_aug_opt.brightness else 0
            random_gamma = np.random.uniform(0.7, 1.2, 1)[0] if photo_aug_opt.gamma else 0
            random_contrast = np.random.uniform(0.7, 1.2, 1)[0] if photo_aug_opt.contrast else 0
            random_light = np.random.uniform(0.5, 5.0, 1)[0] if photo_aug_opt.light else 0
            augmentations = [transforms.ToPIL(np.uint8)]
            augmentations.append(transforms.PhotometricAug(random_brightness, random_gamma, random_contrast))
            augmentations.append(transforms.ToTensor())
            augmentations.append(transforms.Lighting(random_light))
            augmentations.append(transforms.Normalizer())
        else:
            augmentations = [transforms.ToTensor(), transforms.Normalizer()]
            
        last_transform = transforms.Compose(augmentations)
        inputs, targets = last_transform(inputs, targets)
        
        # check nan values
        error_handler(self.check_inf_nan(inputs), 'invalid value found in inputs', __name__, True)
        error_handler(self.check_inf_nan(targets), 'invalid value found in targets', __name__, True)

        return inputs, targets


class raw_transform(object):

    def __init__(self, option, crop=False):
        self.option = option
        self.crop = crop

    def get_size(self, inputs, ratio, factor):

        if inputs is None:
            raise NotImplementedError('inputs must not be None')
        size = None
        for input in inputs:
            if input is not None:
                size = (input.shape[0], input.shape[1])
                break

        if size is None:
            raise NotImplementedError('size must not be None')

        n = np.ceil(np.array(size) * ratio // factor).astype('int')

        return tuple(factor * n)

    def apply(self, inputs_, targets_):
        
        inputs, targets = transforms.ToNumpy()(inputs_.copy(), targets_.copy())

        if self.crop:
            augment_opt = self.opt.crop_aug
            if augment_opt.crop_aug.type == 'soft_crop':
                cropped_size = self.get_size(inputs, crop_aug_opt.soft_crop.crop_ratio, augment_opt.crop_aug.soft_crop.crop_factor)
            else:
                cropped_size = (augment_opt.crop_aug.hard_crop.crop_height, augment_opt.crop_aug.hard_crop.crop_width)
            crop_aug = basic_transforms.Cropper(self.option, cropped_size)
            inputs, targets = crop_aug(inputs, targets)

        return transforms.ToTensor()(inputs, targets)