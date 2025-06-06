from argparse import ArgumentParser
from models.lightning_model import getModel
from fins_lightning_dataloader import DareDataModule
import torch as t
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.profilers import AdvancedProfiler
from pytorch_lightning import loggers as pl_loggers
from utils.utils import load_config
from utils.progress_bar import getProgressBar
import random
import numpy as np
import os
from WaveUnet.waveunet import Waveunet
from pytorch_lightning.loggers import TensorBoardLogger



def main(args):
    # ===========================================================
    # Configuration
    cfg = load_config(args.config_path)

    random.seed(   cfg['random_seed'])
    np.random.seed(cfg['random_seed'])
    t.manual_seed(cfg['random_seed'])


    channels         = cfg['WaveUnet']['channels']
    kernel_size_down = cfg['WaveUnet']['kernel_size_down']
    kernel_size_up   = cfg['WaveUnet']['kernel_size_up']
    levels           = cfg['WaveUnet']['levels']
    feature_growth   = cfg['WaveUnet']['feature_growth']
    output_size      = cfg['WaveUnet']['output_size']
    sr               = cfg['WaveUnet']['sr']
    conv_type        = cfg['WaveUnet']['conv_type']
    res              = cfg['WaveUnet']['res']
    features         = cfg['WaveUnet']['features']
    instruments      = ["speech", "rir"]
    num_features     = [features*i for i in range(1, levels+1)] if feature_growth == "add" else \
                        [features*2**i for i in range(0, levels)]
    target_outputs   = int(output_size * sr)
    learning_rate    = cfg['WaveUnet']['learning_rate']
    model            = Waveunet(channels, num_features, channels, instruments, kernel_size_down=kernel_size_down, kernel_size_up=kernel_size_up, target_output_size=target_outputs, conv_type=conv_type, res=res, separate=False, learning_rate=learning_rate, 
                        config = cfg)

    print("Using model " + model.name)
    # Data Module
    datamodule = DareDataModule(config=cfg)

    # Checkpoints
    ckpt_callback = ModelCheckpoint(
        **cfg['ModelCheckpoint'],
        filename = model.name + "-{epoch:02d}-{val_loss:.4f}",
        save_top_k=-1
    )
    
    # Learning Rate Monitor
    lr_monitor = LearningRateMonitor(**cfg['LearningRateMonitor'])

    # # Strategy
    # strategy = DDPStrategy(**cfg['DDPStrategy'])

    # Profiler
    profiler = AdvancedProfiler(**cfg['AdvancedProfiler'])

    logger = TensorBoardLogger("wavenet_logs")
    
    # PyTorch Lightning Train
    trainer = pl.Trainer(
        **cfg['Trainer'],
        # strategy=strategy,
        #profiler=profiler,
        logger = logger,
        callbacks=[ckpt_callback,lr_monitor]
        )

    trainer.fit(
        model      = model,
        datamodule = datamodule, 
        ckpt_path  = args.ckpt_path
    )

    # ===========================================================
    # PyTorch Lightning Test
    trainer.test(
        model      = model,
        datamodule = datamodule,
        ckpt_path  = "best"
        )
    
    return True

if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("--config_path", type=str, default="config.yaml", 
        help="A full or relative path to a configuration yaml file. (default: config.yaml)")
    parser.add_argument("--ckpt_path", type=str, default=None,
        help="A full or relative path to a checkpoint file. (default: None)")
        
    args = parser.parse_args()
    main(args)