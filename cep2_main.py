from argparse import ArgumentParser
from models.cep3_model import Test
from fins_lightning_dataloader import DareDataModule
import torch as t
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from utils.utils import load_config
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

    model = Test(learning_rate = cfg.dare.learning_rate,
                    nwins = cfg.nwins,
                    use_transformer = cfg.dare.use_transformer,
                    alphas = cfg.dare.alphas,
                    softargmax_beta = cfg.dare.softargmax_beta,
                    residual=cfg.dare.residual,
                    delays = cfg.Encoding.delays,
                    win_size = cfg.Encoding.win_size,
                    cutoff_freq = cfg.Encoding.cutoff_freq,
                    sample_rate = cfg.sample_rate,
                    plot_every_n_steps=cfg.plot_every_n_steps,
                    norm_cepstra = cfg.dare.norm_cepstra,
                    cepstrum_target_region = cfg.dare.cep_target_region)

    # Data Module
    datamodule = DareDataModule(config=cfg)

    # Checkpoints
    ckpt_callback = ModelCheckpoint(
        **cfg['ModelCheckpoint'],
        filename = "-{epoch:02d}-{step}-{val_loss:.2f}",
    )
    
    # create custom logger to allow fig logging
    tensorboard = pl_loggers.TensorBoardLogger('./')

    # PyTorch Lightning Train
    trainer = pl.Trainer(
        **cfg['Trainer'],
        callbacks=[ckpt_callback],
        logger = tensorboard
    )

    trainer.fit(
        model      = model,
        datamodule = datamodule,
        ckpt_path = args.ckpt_path
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