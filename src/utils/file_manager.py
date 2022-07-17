import pdb

import shutil
import numpy as np
from pathlib import Path
import torch
import logging


def make_iterative_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)
    return wrapper

def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret
    return wrapper

@make_iterative_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type for tensor2float")

@make_iterative_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.cpu().numpy()
    else:
        raise NotImplementedError("invalid input type for tensor2numpy")


def test_wrong_tensor(x):
    condition = torch.isnan(x) | torch.isinf(x)
    if torch.sum(condition) > 0:
        return True
    else:
        return False
    
def error_handler(condition, expression, name, stop=False):
    
    '''
    :param condition: condition to check
    :param expression: error message
    :param name: location of error, use __name__ in this place
    :param stop: if condition is wrong, stop the process
    :return:
    '''

    try:
        assert condition
    except:
        if stop:
            raise NotImplementedError('%s : %s\n' % (name, expression))
        else:
            print('%s : %s\n' % (name, expression))
    
# create directory
def makedir_custom(path, opt=False):
    '''
    :param path: src path (string)
    :param opt: if exists, overwrite or not
    :return:
    '''

    if not isinstance(path, Path):
        path = Path(path)

    if opt and path.is_dir():
        try:
            path.rmdir()
        except:
            shutil.rmtree(str(path))

    if not path.is_dir():
        path.mkdir()

    return path

# move directory
def movedir_custom(srcpath, dstpath, opt=False):

    error_handler(Path(srcpath).is_dir(), '%s is invalid, please check!' % srcpath, __name__, False)

    if opt:
        shutil.copytree(srcpath, dstpath)
    else:
        shutil.move(srcpath, dstpath)

    return dstpath
    
    
# logging setup
def setup_logger(logger_name, log_file, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(fileHandler)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    return logger

