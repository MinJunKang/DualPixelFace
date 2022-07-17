
import pdb
import torch
from runpy import run_path
from pathlib import Path


def model_selector(option):
    
    # model selection
    loaded_file = run_path(str(Path('src/model') / option.model_name / 'mainmodel.py'))
    if loaded_file is None:
        raise NotImplementedError('You should check if model main class name (upper case) is the same as your modelname !')
    else:
        model = loaded_file[option.model_name.upper()](option)
    
    # load checkpoint's model parameters (test or demo mode)
    if option.load_model is not None and option.mode is not 'train':
        checkpoint = torch.load(option.load_model)
        if 'state_dict' in checkpoint:
            loaded_checkpoints = checkpoint['state_dict']
        elif 'model' in checkpoint:
            loaded_checkpoints = checkpoint['model']
        model.load_state_dict(loaded_checkpoints, strict=option.load_strict)
    
    return model


def optimizer_selector(params, option):
    
    if option.optim == 'adam':
        optimizer = torch.optim.Adam(params, lr=float(option.init_lr), betas=(0.9, 0.999), eps=1e-5)
    elif option.optim == 'sgd':
        optimizer = torch.optim.SGD(params, lr=float(option.init_lr), momentum=0.9, weight_decay=2e-4)
    elif option.optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, lr=float(option.init_lr), eps=1e-5)
    else:
        raise NotImplementedError('optimizer is not defined, please check your optimizer configuration !')

    return optimizer


def scheduler_selector(optimizer, option):
    
    if option.scheduler == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.5)  # step_size, gamma
    elif option.scheduler == 'explr':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)  # gamma
    elif option.scheduler == 'cosanneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500, 1e-6)  # T_max (iterations), eta_min (minimum learning rate)
    elif option.scheduler == 'none':
        scheduler = None
    else:
        raise NotImplementedError('scheduler is not defined, please check your scheduler configuration !')

    return scheduler