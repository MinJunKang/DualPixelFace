
import pdb
import torch
import numpy as np
from texttable import Texttable
from src.utils.file_manager import error_handler, tensor2numpy
from src.metric.normal_dp.metric import calNormalAcc, calNormalAccRMSE


'''
    Functions for logging benchmark results
    normal benchmark : n_err_mean
'''


class normal_dp_Benchmark(object):
    
    def __init__(self, option, samplenum=-1):
        self.opt = option
        self.t = Texttable()
        self.t.set_deco(Texttable.HEADER)
        self.samplenum = samplenum
        self.index = 0
        self.metric = dict()

        self.metric['n_err_mean'] = []
        self.metric['n_err_rmse'] = []

    def measure(self, pred_, batch, log=True):
        
        pred = pred_['pred_normal']
        target = batch['normal']
        mask = batch['mask'] if 'mask' in batch.keys() else torch.ones_like(pred[:, 0, :, :])

        data = [tensor2numpy(calNormalAcc(pred[:, 0, :, :], target, mask.unsqueeze(1))),
                tensor2numpy(calNormalAccRMSE(pred[:, 0, :, :], target, mask.unsqueeze(1)))]

        if log:
            self.update(data)

        return data

    def update(self, data):

        if (self.samplenum != -1) and (self.index >= self.samplenum):
            return

        error_handler(len(data) == len(self.metric.keys()), "data should have %d elements" % len(self.metric.keys()),
                      __name__, True)

        for idx, key in enumerate(self.metric.keys()):
            self.metric[key].append(data[idx])

        self.index += 1

    def get_value(self, pos=-1, use_chart=False):

        results = []

        if self.index == 0:
            print('No data stored')
            return None, None

        if pos == -1:
            for key in self.metric.keys():
                results.append(np.asarray(self.metric[key]).mean())
        else:
            for key in self.metric.keys():
                results.append(self.metric[key][pos])

        if use_chart:
            t = self.t
            t.set_deco(Texttable.HEADER)
            t.set_cols_dtype(['f' for key in self.metric.keys()])
            t.set_cols_width([10 for key in self.metric.keys()])
            t.add_row([key for key in self.metric.keys()])
            t.add_rows([results], header=False)
            t.set_cols_align(['r' for key in self.metric.keys()])
            return results, t
        else:
            return results

    def clear(self):

        for key in self.metric.keys():
            self.metric[key] = []
