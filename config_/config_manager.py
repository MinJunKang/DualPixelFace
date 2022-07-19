
import pdb
import os
import json
from pathlib import Path
from src.utils.file_manager import makedir_custom, error_handler, setup_logger

class obj(object):
    def __init__(self, d):
        for key, value in d.items():
            if isinstance(value, (list, tuple)):
                setattr(self, key, [obj(x) if isinstance(x, dict) else x for x in value])
            else:
                setattr(self, key, obj(value) if isinstance(value, dict) else value)
                

class Configuration(object):
    def __init__(self, args):
        
        self.data = {'model': {}, 'dataset': {}}
        
        # config from argparse
        self.args = args
        self.config = args.config
        self.workspace = args.workspace
        self.load_model = None if args.load_model is None else Path(args.load_model)
        
        # check validity
        self.config_path = Path('config_') / (self.config + '.json')
        error_handler(self.config_path.is_file(), "invalid config name", __name__, True)
        if self.load_model is not None:
            error_handler(self.config_path.is_file(), "invalid config name", __name__, True)
            self.data.update({'load_model': str(self.load_model.absolute())})
        else:
            self.data.update({'load_model': None})
            
        self.initialize()
        
    def option_check(self, value, options=None):
        error_handler(value in options, "option_check failed : %s" % value, __name__, True)
        
    def range_check(self, value, min=None, max=None):
        if min is not None:
            error_handler(value >= min, "range_check failed", __name__, True)
        if max is not None:
            error_handler(value <= max, "range_check failed", __name__, True)
            
    def load_json_data(self, path):
        with open(str(path)) as file:
            config_data = json.load(file)
        return config_data
            
    def initialize(self):
        
        # load config
        self.data.update(self.load_json_data(self.config_path))
        self.data['sync_batch'] = True if self.data['accelerator'] == 'ddp' else False
        root_path = makedir_custom('./workspace')
        model_path = makedir_custom(root_path / self.data['model_name'])
        workspace_path = makedir_custom(model_path / self.workspace)
        logger_path = makedir_custom(workspace_path / 'log', self.data['mode'] == 'train')
        output_path = makedir_custom(workspace_path / 'output', True)
        logger_text = setup_logger('train_log', str(output_path / 'log_text.txt'))
        logger_text.info(self.args)
        for key, value in vars(self.args).items():
            logger_text.info(key + ': ' + str(value))
        self.data.update({'model_path': model_path, 'workspace_path': workspace_path, 'logger_path': logger_path, 'output_path': output_path})
            
        # load model config
        model_config_path = Path('src/model') / self.data['model_name'] / (self.data['model_config'] + '.json')
        error_handler(model_config_path.is_file(), "invalid model config", __name__, True)
        self.data['model'] = self.load_json_data(model_config_path)
        
        # load dataset config
        data_config_path = Path('dataloader') / self.data['dataset_name'] / (self.data['dataset_config'] + '.json')
        error_handler(data_config_path.is_file(), "invalid dataset config", __name__, True)
        self.data['dataset'] = self.load_json_data(data_config_path)
        
        # augmentation if needed
        if 'augmentation' in self.data.keys():
            preprocess_option = self.load_json_data(Path('dataloader/preprocess') / (self.data['augmentation_config'] + '.json'))
            for aug in self.data['augmentation']:
                if aug in preprocess_option.keys():
                    self.data[aug] = preprocess_option[aug]
        
        # option check for validation
        
    def update(self, config):
        if config is not None:
            self.data.update(config)
        
        # option check for validation
        
    def get_config(self):
        return obj(self.data)