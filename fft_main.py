from argparse import ArgumentParser
from models.fft_lightning_model import getModel
from datasets.fft_reverb_speech_data import DareDataModule
import torch as t
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.profilers import AdvancedProfiler
from pytorch_lightning import loggers as pl_loggers
from utils.utils import getConfig
from utils.progress_bar import getProgressBar
import random
import numpy as np
import os
os.environ['MASTER_ADDR'] = str(os.environ.get('HOST', '::1'))

random.seed(   getConfig()['random_seed'])
np.random.seed(getConfig()['random_seed'])
t.manual_seed( getConfig()['random_seed'])

def main(args):
    # ===========================================================
    # Configuration
    cfg = getConfig(config_path=args.config_path)

    # PyTorch Lightning Models
    model = getModel(**cfg['Model'], 
                        delays = cfg["Encoding"]["delays"],
                        win_size = cfg["Encoding"]["win_size"], 
                        cutoff_freq = cfg["Encoding"]["cutoff_freq"],
                        sample_rate = cfg["sample_rate"], 
                        plot_every_n_steps=cfg["plot_every_n_steps"],
                        fft_target_region = cfg["fft_target_region"])

    # Data Module
    datamodule = DareDataModule(config=cfg)

    # Checkpoints
    ckpt_callback = ModelCheckpoint(
        **cfg['ModelCheckpoint'],
        filename = model.name + "-{epoch:02d}-{val_loss:.2f}",
    )
    
    # Learning Rate Monitor
    lr_monitor = LearningRateMonitor(**cfg['LearningRateMonitor'])

    # Strategy
    strategy = DDPStrategy(**cfg['DDPStrategy']) # changed from DataParallelStrategy due to deprecation

    # Profiler
    profiler = AdvancedProfiler(**cfg['AdvancedProfiler'])

    # create custom logger to allow fig logging
    tensorboard = pl_loggers.TensorBoardLogger('./')

    # PyTorch Lightning Train
    trainer = pl.Trainer(
        **cfg['Trainer'],
        # strategy=strategy,
        profiler=profiler,
        callbacks=[ckpt_callback,lr_monitor],
        logger = tensorboard
    )

    trainer.fit(
        model      = model,
        datamodule = datamodule
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
        
    args = parser.parse_args()
    main(args)