from argparse import ArgumentParser
from fins_lightning_dataloader import DareDataModule
import torch as t
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.profilers import AdvancedProfiler
from pytorch_lightning import loggers as pl_loggers
from models.joint_model import JointModel
from utils.utils import load_config

import argparse
import random
import numpy as np
import os
os.environ['MASTER_ADDR'] = str(os.environ.get('HOST', '::1'))


def main(args):
    # ===========================================================
    # Configuration
    cfg = load_config(args.config_path)

    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    t.manual_seed(cfg.random_seed)

    # Data Module
    datamodule = DareDataModule(config=cfg)

    # Checkpoints
    ckpt_callback = ModelCheckpoint(
        **cfg['ModelCheckpoint'],
        filename = "-{epoch:02d}-{step}-{val_loss:.2f}",
    )

 
    # create custom logger to allow fig logging
    tensorboard = pl_loggers.TensorBoardLogger('./')

    fins_config = load_config(args.config_path)

    model = JointModel(fins_config)

    # PyTorch Lightning Train
    trainer = pl.Trainer(
        gradient_clip_val = fins_config.fins.gradient_clip_value,
        callbacks = [ckpt_callback],
        logger = tensorboard,
        # strategy = DDPStrategy(process_group_backend="gloo"),
        **cfg['Trainer']
    )

    trainer.fit(
        model      = model,
        datamodule = datamodule,
        ckpt_path = args.ckpt_path
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("--config_path", type=str, default="config.yaml", 
        help="A full or relative path to a configuration yaml file. (default: config.yaml)")
    parser.add_argument("--ckpt_path", type=str, default=None,
        help="A full or relative path to a checkpoint file. (default: None)")
        
    args = parser.parse_args()
    main(args)