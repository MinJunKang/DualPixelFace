from src.utils.file_manager import error_handler
from src.utils.geometry import intrinsic2KD
from PIL import Image
from pathlib import Path
import numpy as np
import random
import json
import cv2
import pdb

'''
Path reader for RCV DPLoader
'''


class RCV_DPreader(object):
    '''
    RCV Facial Dataset Loader for DUAL PIXEL
    '''

    def __init__(self, option, parentdir, training):

        self.option = option
        self.training = training
        self.parentdir = Path(parentdir)
        self.abvalue_list = {1: [-26996.48848727, 32.984822], 2: [-25727.48737484, 31.80317696], 
                             3: [-24940.24188275, 30.52371982], 4: [-25821.86619949, 32.03359466], 
                             5: [-26735.69581971, 33.24327157], 6: [-22694.45143825, 27.76217617], 
                             7: [-23598.82548605, 29.1246567], 8: [-26482.94764346, 32.91372342]}

    def read_directory(self):
        '''
        Task : read train.txt / test.txt to get parentdir information
        :return:
        '''

        pathdata = []

        if self.training:
            filepath = self.parentdir / 'train.txt'
        else:
            filepath = self.parentdir / 'test.txt'
        error_handler(filepath.is_file(), '%s does not exist.' % str(filepath), True)

        with open(str(filepath), 'r') as f:
            while True:
                line = f.readline()
                if not line: break
                line = line.replace('\n', '').replace('\r', '')
                pathdata.append(self.parentdir / line)

        return pathdata

    def read_rcv_path(self):
        '''
        Task : read json information
        :return:
        '''

        jsonpaths = []
        count = 0

        pathdata_all = self.read_directory()  # read datapath for DP, path object
        data_option = self.option.dataset  # dataset's option

        for path in pathdata_all:
            path_ = path / 'JSON'

            for jsonpath in path_.glob('*.json'):
                multiview_path = dict()
                jsonfullpath = path_ / jsonpath
                with open(str(jsonfullpath)) as json_file:
                    json_data = json.load(json_file)
                    # invalid dataset
                    if not bool(json_data['INFO']['valid']):
                        continue
                    # unwanted light condition
                    if not json_data['INFO']['lightidx'] in data_option.light:
                        continue
                    # unwanted gender condition
                    if not json_data['INFO']['gender'] in data_option.gender:
                        continue
                    # unwanted view condition
                    if not json_data['INFO']['camidx'] in data_option.viewpoint:
                        continue
                    # unwanted expression
                    if not json_data['INFO']['expression'] in data_option.expression:
                        continue
                    # unwanted position
                    if not json_data['INFO']['position'] in data_option.distance:
                        continue
                    # unwanted direction
                    if not json_data['INFO']['direction'] in data_option.direction:
                        continue
                    if self.option.use_multi:
                        subpaths = []
                        viewidx = int(np.char.split(str(jsonpath), '_').tolist()[-2])
                        lightidx = int(json_data['INFO']['lightidx'])
                        for cam in data_option.select_view:
                            filename = 'INFO_%d_%d_%d.json' % (cam, viewidx, lightidx)
                            filepath = str(path_ / filename)
                            with open(filepath) as sub_json_file:
                                sub_json_data = json.load(sub_json_file)
                                if not bool(sub_json_data['INFO']['valid']):
                                    continue
                            subpaths.append(filepath)
                        # if the number is insufficient, use dummy
                        if len(subpaths) != len(data_option.select_view):
                            for i in range(len(data_option.select_view) - len(subpaths)):
                                subpaths.append(subpaths[-1])
                        if len(subpaths) > 0:
                            multiview_path['ref_view'] = subpaths
                            multiview_path['tar_view'] = str(jsonfullpath)
                            multiview_path['parentdir'] = str(path)
                            jsonpaths.append(multiview_path)
                    else:
                        multiview_path['ref_view'] = None
                        multiview_path['tar_view'] = str(jsonfullpath)
                        multiview_path['parentdir'] = str(path)
                        jsonpaths.append(multiview_path)
                    count += 1
                    print('reading path : %d' % count)

        return jsonpaths, len(jsonpaths)

    def check_inf_nan(self, tensors):
        for tensor in tensors:
            if tensor is not None:
                if (np.sum(np.isinf(tensor)) > 0) or (np.sum(np.isnan(tensor)) > 0):
                    return False
        return True

    def read_img(self, json_data, rootdir):

        # read img
        leftpath = rootdir / json_data['PATH']['left']
        rightpath = rootdir / json_data['PATH']['right']
        lrpath = rootdir / json_data['PATH']['lrsum']

        # read images
        leftimg = Image.open(str(leftpath))
        rightimg = Image.open(str(rightpath))
        lrimg = Image.open(str(lrpath))

        return leftimg, rightimg, lrimg

    def read_depth(self, json_data, rootdir):

        # read depth
        depthpath = rootdir / json_data['PATH']['depth']
        depth = np.load(str(depthpath))

        # read mask
        if 'mask' in json_data['PATH'].keys():
            mask = np.load(str(rootdir / json_data['PATH']['mask'])) > 0
        else:
            mask = depth > 0

        # clip the depth value
        max_depth = np.max(depth[mask])
        idepth = np.divide(max_depth, depth, where=mask)
        idepth[~mask] = 0.0
        depth[~mask] = 0.0

        return depth, idepth, mask

    def read_normal(self, json_data, rootdir):

        # read normal map
        normalpath = rootdir / json_data['PATH']['normal']

        normal = np.load(str(normalpath))

        # mask
        normal = np.asarray(normal, dtype=np.float32)
        mask = cv2.cvtColor(normal, cv2.COLOR_BGR2GRAY) > 0

        return normal, mask

    def read_albedo(self, json_data, rootdir):

        # read normal map
        albedopath = rootdir / json_data['PATH']['albedo']

        albedo = np.load(str(albedopath))

        # mask
        albedo = np.asarray(albedo, dtype=np.float32)
        mask = albedo > 0

        return albedo, mask

    def read_disparity(self, json_data, parentdir, abvalue=None, metadata=None, fy=None):

        # read depth
        depth, idepth, mask = self.read_depth(json_data, parentdir)

        # depth to disparity conversion
        if abvalue != None:
            # if there is pre-calibrated abvalue, use it!
            disparity = np.add(np.divide(abvalue[0], depth, where=mask, dtype='float64'),
                               abvalue[1], where=mask, dtype='float64')
            abvalue = [abvalue[1], abvalue[0]]
        elif metadata != None:
            # if there is not pre-calibrated abvalue, calculate it from given metadata
            f = metadata[0]
            g = metadata[1]
            fnum = metadata[2]
            t = metadata[3] * 0.001  # (mm) scale
            k = 0.13372  # should be accurately calibrated, but use this rough value.

            # refer to https://www.edmundoptics.co.kr/knowledge-center/application-notes/imaging/lens-iris-aperture-setting/
            # refer to https://www.optowiki.info/glossary/working-f-number/
            # refer to https://www.edmundoptics.co.kr/knowledge-center/application-notes/imaging/6-fundamental-parameters-of-an-imaging-system/
            
            # get a, b from pre calibrated parameters : a is -23971.92 and b is 24.71332 for gvalue=970
            a = -k * (fy / fnum) * f / (1 - f / g)
            b = k * (fy / fnum) * f / (1 - f / g) * (1 / g)
                
            abvalue = [b, a]
            disparity = np.add(np.divide(abvalue[1], depth, where=mask, dtype='float64'),
                               abvalue[0], where=mask, dtype='float64')
        else:
            raise NotImplementedError('There is no way to convert depth to disparity!')

        # clip the disparity value
        disparity[~mask] = np.max(disparity[mask]) * 50.0
        disparity[np.isnan(disparity)] = np.max(disparity[mask]) * 50.0
        disparity[np.isinf(disparity)] = np.max(disparity[mask]) * 50.0

        # check value error
        error_handler(self.check_inf_nan(disparity), 'Nan or inf value is detected in disparity map', __name__, True)

        return disparity, depth, idepth, mask, abvalue

    def read_calib(self, json_data):

        # read intrinsic
        strvalue = json_data['PARAMS']['intrinsic']
        intrinsic = eval(strvalue[6:-1])

        # read extrinsic
        strvalue = json_data['PARAMS']['pose']
        extrinsic = eval(strvalue[6:-1])

        # read calibrated light direction value
        strvalue = json_data['PARAMS']['Lvalue']
        if strvalue is not None:
            L = eval(strvalue[6:-1])
        else:
            L = None

        # read abvalue
        # strvalue = json_data['PARAMS']['abvalue']
        # abvalue = eval(strvalue[6:-1])
        abvalue = self.abvalue_list[json_data['INFO']['camidx']]

        # meta data
        metadata = [135.0, 970.0, 5.657, 5.36]  # (focal length(mm), focused distance(mm), fnum, pixel size(um))

        return intrinsic, extrinsic, L, abvalue, metadata

    def load_data_depth(self, json_data, parentdir, multi=False):
        '''
        Task : load all the data for depth prediction task
        :param multi:
        :param json_data:
        :return:
        '''

        # read calibration data
        intrinsic, extrinsic, L, abvalue, metadata = self.read_calib(json_data)
        
        # convert extrinsic and intrinsic to P, K matrix
        P = np.reshape(np.transpose(np.concatenate(
            (np.expand_dims(np.array(extrinsic), -1), np.zeros((3, 1)), np.ones((1, 1))), axis=0)), [4, 4])  # [4, 4]
        K, _ = intrinsic2KD(intrinsic)

        # read images
        right, left, lr = self.read_img(json_data, parentdir)  # should be changed if the dataset is modified (left, right, lr)

        # read normal if necessary
        if (~multi & self.option.use_normal) | (multi & self.option.multi_view.use_normal):
            normal, _ = self.read_normal(json_data, parentdir)
            normal = np.ascontiguousarray(normal, dtype=np.float32)
        else:
            normal = None

        # read albedo if necessary
        if (~multi & self.option.use_albedo) | (multi & self.option.multi_view.use_albedo):
            albedo, _ = self.read_albedo(json_data, parentdir)
            albedo = np.ascontiguousarray(albedo, dtype=np.float32)
        else:
            albedo = None

        # convert metric depth to defocus-disparity
        dispmap, depthmap, idepthmap, mask, abvalue = self.read_disparity(json_data, parentdir, abvalue=abvalue, metadata=metadata, fy=K[1][1])  # if fy is None (no use of bias)
        
        # convert as contiguous array
        mask = np.ascontiguousarray(mask, dtype=np.float32)
        dispmap = np.ascontiguousarray(dispmap, dtype=np.float32)
        depthmap = np.ascontiguousarray(depthmap, dtype=np.float32)

        # output list check
        if multi:
            if not self.option.multi_view.use_dual_pixel:
                left = None
                right = None
            if not self.option.multi_view.use_center_img:
                lr = None
            if not self.option.multi_view.use_mask:
                mask = None
            if not self.option.multi_view.use_disparity:
                dispmap = None
            if not self.option.multi_view.use_depth:
                depthmap = None
            if not self.option.multi_view.use_idepth:
                idepthmap = None
            if not self.option.multi_view.use_normal:
                normal = None
            if not self.option.multi_view.use_albedo:
                albedo = None
        else:
            if not self.option.use_dual_pixel:
                left = None
                right = None
            if not self.option.use_center_img:
                lr = None
            if not self.option.use_mask:
                mask = None
            if not self.option.use_disparity:
                dispmap = None
            if not self.option.use_depth:
                depthmap = None
            if not self.option.use_idepth:
                idepthmap = None
            if not self.option.use_normal:
                normal = None
            if not self.option.use_albedo:
                albedo = None

        if L is None:
            try:
                L = np.zeros((3, 3))  # dummy
            except ...:
                L = np.zeros(3)  # dummy

        inputs = [left, right, lr]
        targets = [depthmap, mask, dispmap, idepthmap, normal, albedo]
        params = [np.float32(K), np.float32(P), np.float32(abvalue), np.float32(metadata), None]

        return inputs, targets, params