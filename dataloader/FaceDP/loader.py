

import os
import pdb
import json
import torch
import numpy as np
from pathlib import Path

from src.utils.file_manager import error_handler

from dataloader.FaceDP.path_reader import RCV_DPreader
import dataloader.preprocess.preprocess as preprocess


'''
RCV dual pixel dataset format

There are 2 task : train / test

There are 8 position, 3 facial expression, 2 different distance information : 

Read JSON file, file contains all the path information of imgs, depths, normals, albedos and calibration informations

File format : 

    ALBEDO : ALBEDO_#cam_#view

    CALIBRATION : 
        Disp2DEpth.npy
        intrinsic.npy
        light.npy
        pose.npy

    DEPTH : 
        DEPTH_#cam_#view.npy

    IMG : 
        LEFT : IMG_#cam_#view_#light.JPG
        RIGHT : IMG_#cam_#view_#light.JPG
        LRSUM : IMG_#cam_#view_#light.JPG

    JSON : 
        INFO_#cam_#view_#light.json
        contents : 
            INFO : {
                valid : there is invalid scene (low light condition, or double captured)
                object : parent dir name
                gender : person's gender
                camidx : captured camera idx
                lightidx : light idx
                expression : expression type
                position : two distances available (forward, backward)
                direction : four head directions
            }
            PATH : {
                root : path information
                left : path information
                right : path information
                lrsum : path information
                depth : path information
                normal : path information
                albedo : path information
                calibration : path information
            }
            PARAMS : {
                intrinsic : [3, 3] array - intrinsics
                pose : [12, 1] array - extrinsics
                Lvalue : [3, 1] array or null value (projector 1 case) - light direction vector
                abvalue : [2, 1] array - affine parameter from metric depth to disparity
            }

    NORMAL : 
        NORMAL_#cam_#view.npy


'''


class FaceDPLoader(torch.utils.data.Dataset):

    def __init__(self, option, training):

        self.opt = option  # options (config.py)
        self.training = training  # current mode
        self.parentdir = option.dataset.path
        self.use_multi = option.use_multi  # use photometric consistency loss as multi-view dataset

        # check conditions to proceed
        error_handler(os.path.isdir(self.parentdir), '%s does not exist.', __name__, True)

        # create npy path file for convenience
        if self.training:
            if self.use_multi:
                path_saved = self.opt.dataset_name + '_train_multi.npy'
            else:
                path_saved = self.opt.dataset_name + '_train_single.npy'
        else:
            if self.use_multi:
                path_saved = self.opt.dataset_name + '_test_multi.npy'
            else:
                path_saved = self.opt.dataset_name + '_test_single.npy'

        # Read JSON
        self.pathreader = RCV_DPreader(option, self.parentdir, self.training)
        if not os.path.isfile(path_saved):
            self.pathdata, self.datalen = self.pathreader.read_rcv_path()
            np.save(path_saved, [self.pathdata, self.datalen])
        else:
            self.pathdata, self.datalen = np.load(path_saved, allow_pickle=True)

        # Transform
        self.transform = preprocess.basic_transform(self.opt)
        self.raw_transform = preprocess.raw_transform(self.opt)

    def transpose_list(self, lists):
        return [list(x) for x in zip(*lists)]

    def add2output(self, sample_out, tensors, names):
        assert (len(tensors) == len(names))
        for idx, name in enumerate(names):
            if isinstance(tensors[idx], list):
                if isinstance(tensors[idx][0], torch.Tensor):
                    sample_out[name] = torch.cat(tensors[idx], dim=0)
                elif tensors[idx][0] is not None:
                    sample_out[name] = np.asarray(tensors[idx])
            elif tensors[idx] is not None:
                sample_out[name] = tensors[idx]
        return sample_out

    def __getitem__(self, index):
        
        sample_out = dict()
        json_path = self.pathdata[index]
        parent_dir_ = Path(json_path['parentdir'])

        # single view case
        with open(json_path['tar_view']) as json_file:
            json_data = json.load(json_file)
        inputs, targets, params = self.pathreader.load_data_depth(json_data, parent_dir_)

        # Transform for single view input
        processed_inputs, processed_targets = self.transform.apply(inputs, targets)

        # add cropped starting coordinate to params, if no crop_aug will be set to [0, 0]
        params.append(self.transform.coords)

        # add to sample_out
        sample_out = self.add2output(sample_out, processed_inputs, ['left', 'right', 'center'])
        sample_out = self.add2output(sample_out, processed_targets, ['depth', 'mask', 'disp', 'idepth', 'normal', 'albedo'])
        sample_out = self.add2output(sample_out, params, ['K', 'P', 'abvalue', 'metadata', 'L', 'coords'])

        # adjust K value by cropped coordinates
        sample_out['K'][0, 2] = sample_out['K'][0, 2] - self.transform.coords[0]
        sample_out['K'][1, 2] = sample_out['K'][1, 2] - self.transform.coords[1]

        # If we use raw data (no data augmentation)
        if self.opt.use_raw:
            raw_inputs, raw_targets = self.raw_transform.apply(inputs, targets)
            sample_out = self.add2output(sample_out, raw_inputs, ['raw_left', 'raw_right', 'raw_center'])
            sample_out = self.add2output(sample_out, raw_targets, ['raw_depth', 'raw_mask', 'raw_disp', 'raw_idepth',
                                                                   'raw_normal', 'raw_albedo'])
        # If we use multi-view data (no data augmentation)
        if self.use_multi:
            error_handler(json_path['ref_view'] is not None, __name__, 'multi-view dataloader error')

            inputs_multi, targets_multi, params_multi = [], [], []
            for json_path_multi in json_path['ref_view']:

                with open(json_path_multi) as json_file:
                    json_data = json.load(json_file)

                # read as list of array
                inputs_, targets_, params_ = self.pathreader.load_data_depth(json_data, parent_dir_, True)
                inputs_, targets_ = self.raw_transform.apply(inputs_, targets_)

                inputs_multi.append(inputs_)
                targets_multi.append(targets_)
                params_multi.append(params_)

            # convert to array type : lists of array or lists of tensors
            inputs_multi = self.transpose_list(inputs_multi)
            targets_multi = self.transpose_list(targets_multi)
            params_multi = self.transpose_list(params_multi)

            sample_out = self.add2output(sample_out, inputs_multi, ['lefts', 'rights', 'centers'])
            sample_out = self.add2output(sample_out, targets_multi, ['depths', 'masks', 'disps', 'idepths', 'normals',
                                                                     'albedos'])
            sample_out = self.add2output(sample_out, params_multi, ['Ks', 'Ps', 'abvalues', 'metadatas', 'Ls'])

        if not self.training:
            groupname = np.char.split(self.pathdata[index]['tar_view'], '/').tolist()[-3]
            sample_out['groupname'] = groupname
        pathname = np.char.split(os.path.split(self.pathdata[index]['tar_view'])[-1], sep='.').tolist()[0]
        sample_out['pathname'] = pathname

        return sample_out

    def __len__(self):
        return len(self.pathdata)