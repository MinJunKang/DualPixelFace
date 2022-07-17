
import pdb

import torch
import argparse
import warnings
warnings.filterwarnings("ignore")
from config_.config_manager import Configuration
from src.model.model_selector import model_selector

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

# parsing arguments
parser = argparse.ArgumentParser(description='Configuration : Dual-Pixel Face Reconstruction')
parser.add_argument('--config', type=str, required=True, help='config to run')
parser.add_argument('--workspace', type=str, required=True, help='workspace name')
parser.add_argument('--load_model', type=str, help='model path to load')
args = Configuration(parser.parse_args())
opt = args.get_config()

def main():
    
    # seed initialize : for reproducibility
    seed_everything(1)
    
    # model selection
    model = model_selector(opt)
    
    # setup logger
    logger = pl_loggers.TensorBoardLogger(str(opt.logger_path))
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor]
    if opt.mode == 'train':
        callbacks.append(ModelCheckpoint(
            dirpath=str(opt.output_path),
            filename='checkpoint_{epoch:02d}',
            save_top_k=-1,
            period=1
        ))
    
    # run the model
    runner = Trainer(
        logger=logger,
        checkpoint_callback=opt.mode == 'train',
        callbacks=callbacks,
        check_val_every_n_epoch=1,
        accelerator=opt.accelerator,
        benchmark=True,
        deterministic=False,
        gpus=torch.cuda.device_count(),
        precision=opt.precision,
        max_epochs=opt.epoch,
        sync_batchnorm=opt.sync_batch,
        amp_level='O2', 
        profiler="pytorch"
    )
    if opt.mode == 'train':
        ckpt_path = opt.load_model if opt.load_model is not None and opt.load_strict else None  # resume training
        runner.fit(model=model, ckpt_path=ckpt_path)
    elif opt.mode == 'test':
        runner.test(model=model, verbose=True)
    else:
        raise NotImplementedError('Wrong mode !!')

    
if __name__ == '__main__':
    main()
    
    
    