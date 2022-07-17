
import pdb
import torch
from pathlib import Path
from runpy import run_path


def loader_selector(option, training):
    
    # loader selection
    loaded_file = run_path(str(Path('dataloader') / option.dataset_name / 'loader.py'))
    if loaded_file is None:
        raise NotImplementedError('dataloader selector : not implemented')
    else:
        loader = loaded_file[option.dataset_name + "Loader"](option, training)
    
    return loader