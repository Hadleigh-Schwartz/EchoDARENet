from argparse import ArgumentParser
from datasets.unified_dataloader import SPDataModule
import torch as t
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from models.joint_model import JointModel
from models.unet_v1 import UNetV1
from models.unet_v2 import UNetV2
from models.unet_v3 import UNetV3
from models.hugging_face import Hugging
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
        "SoundCam": "s",
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
    for rir in cfg.train_rir_datasets:
        if rir in rir_nicknames:
            version += rir_nicknames[rir]
        else:
            raise ValueError(f"Unknown RIR dataset: {rir}")
    version += "[train_rir]"
    for rir in cfg.val_rir_datasets:
        if rir in rir_nicknames:
            version += rir_nicknames[rir]
        else:
            raise ValueError(f"Unknown RIR dataset: {rir}")
    version += "[val_rir]"
    for rir in cfg.test_rir_datasets:
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
    version += "[speech]"

    version += f"_{cfg.min_early_reverb}"
    version += f"_{cfg.align_ir}"
    
    # iterate through version numbers until a unique one is found
    if os.path.exists(f"{log_dir}/{version}"):
        i = 2
        while os.path.exists(f"{log_dir}/{version}_run{i}"):
            i += 1
        version += f"_run{i}"
    
    return version

def main(args):
    # Configuration
    cfg = load_config(args.config_path)

    # Reproducibility
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    t.manual_seed(cfg.random_seed)

    # Data Module
    datamodule = SPDataModule(config=cfg)
    
    if args.model == "unet_v1":
        model = UNetV1(cfg)
    elif args.model == "unet_v3":
        model = UNetV3(cfg)
    elif args.model == "unet_v2":
        model = UNetV2(cfg)
    elif args.model == "joint_v1":
        model = JointModel(cfg)
    elif args.model == "hugging":
        model = Hugging(cfg)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Tensorboard
    log_dir = f"lightning_logs/{args.model}"
    version = create_log_name(cfg, log_dir)
    tensorboard = pl_loggers.TensorBoardLogger('./', name = log_dir, version = version)
   
    # Checkpoints
    ckpt_callback = ModelCheckpoint(
        **cfg['ModelCheckpoint'],
        filename = "{epoch:02d}-{step}",
        dirpath = tensorboard.log_dir
    )

    # Copy the config file to the log directory for future reference. 
    # Have to premptively create the directory, as otheriwse it's not created until fitting begins
    os.makedirs(tensorboard.log_dir, exist_ok=True)
    os.system(f"cp {args.config_path} {tensorboard.log_dir}/config.yaml") # copy config to log dir for reference
    print(f"Configuration file copied to {tensorboard.log_dir}/config.yaml")


    # Initialize Lightning Trainer
    trainer = pl.Trainer(
        gradient_clip_val = cfg.fins.gradient_clip_value,
        callbacks = [ckpt_callback],
        logger = tensorboard,
        # strategy = DDPStrategy(process_group_backend="gloo"), # use this for distributed training
        **cfg['Trainer']
    )

    # Start training
    trainer.fit(
        model      = model,
        datamodule = datamodule,
        ckpt_path = args.ckpt_path
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("--model", type=str, required=True, 
                        help = "Name of the model to train. (Options: unet_v1, unet_v2, unet_v3, unet_v4, ...)")
    parser.add_argument("--config_path", type=str, required=True,
        help="A full or relative path to a configuration yaml file.)")
    parser.add_argument("--ckpt_path", type=str, default=None,
        help="A full or relative path to a checkpoint file. (default: None)")
        
    args = parser.parse_args()
    main(args)