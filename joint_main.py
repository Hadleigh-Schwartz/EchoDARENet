from argparse import ArgumentParser
from datasets.unified_dataloader import DataModule
import torch as t
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from models.joint_model import JointModel
from utils.utils import load_config

import random
import numpy as np
import time
import os
os.environ['MASTER_ADDR'] = str(os.environ.get('HOST', '::1'))

def create_log_name(cfg, log_dir):
    """
    Create a version name based on the configuration file. Include the following information:
    - RIR datasets used
    - Speech datasets used
    - Minimum early reverb
    - Alignment IR
    The version name is used to create a unique directory for the logs and checkpoints.

    Parameters:
        cfg (dict): Configuration dictionary containing the datasets and parameters.
        log_dir (str): Base directory for the logs.

    Returns:
        str: Version string for the log directory.
    """
    # abbreviations for the datasets
    rir_nicknames = {
        "GTU": "g",
        "soundcam": "s",
        "EARS": "e",
        "MIT": "m",
        "homula": "h",
        "sim": "i"
    }
    speech_nicknames = {
        "EARS": "e",
        "LibriSpeech": "l",
        "HiFi": "h"
    }

    # create the version string
    version = ""
    for rir in cfg.rir_datasets:
        if rir in rir_nicknames:
            version += rir_nicknames[rir]
        else:
            raise ValueError(f"Unknown RIR dataset: {rir}")
    version += "_"
    for speech in cfg.speech_datasets:
        if speech in speech_nicknames:
            version += speech_nicknames[speech]
        elif "preenc" in speech:
            version += "p"
        else:
            raise ValueError(f"Unknown speech dataset: {speech}")
    version += f"_{cfg.min_early_reverb}"
    version += f"_{cfg.align_ir}"
    
    # iterate through version numbers until a unique one is found
    if os.path.exists(f"{log_dir}/{version}"):
        i = 2
        while os.path.exists(f"{log_dir}/{version}_v{i}"):
            i += 1
        version += f"_v{i}"
    
    return version

def main(args):
    # Configuration
    cfg = load_config(args.config_path)

    # Reproducibility
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    t.manual_seed(cfg.random_seed)

    # Data Module
    datamodule = DareDataModule(config=cfg)

    model = JointModel(cfg)

    # Tensorboard
    log_dir = "lightning_logs/joint_model"
    version = create_log_name(cfg, log_dir)
    tensorboard = pl_loggers.TensorBoardLogger('./', name = log_dir, version = version)
    # wait until log_dir is created
    while not os.path.exists(tensorboard.log_dir):
        time.sleep(0.1)
        pass
    os.system(f"cp {args.config_path} {tensorboard.log_dir}/config.yaml") # copy config to log dir for reference

    # Checkpoints
    ckpt_callback = ModelCheckpoint(
        **cfg['ModelCheckpoint'],
        filename = "{epoch:02d}-{step}",
        dirpath = tensorboard.log_dir
    )

    # PyTorch Lightning Train
    trainer = pl.Trainer(
        gradient_clip_val = cfg.fins.gradient_clip_value,
        callbacks = [ckpt_callback],
        logger = tensorboard,
        # strategy = DDPStrategy(process_group_backend="gloo"), # use this for distributed training
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