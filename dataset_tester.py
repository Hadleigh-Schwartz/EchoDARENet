"""
For sanity checking dataloaders/datasets.
Replicate the data loading logic of the main training script and make sure I can iterate through batches
and that batch elements look as expected.
"""

from argparse import ArgumentParser
from fins_lightning_dataloader import DareDataModule
from datasets.gtu_rir_data import GTUIRDataset

import torch as t
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

    # # Example testing Data Module
    # datamodule = DareDataModule(config=cfg)
    # # iterate over the dataloader
    # for batch in datamodule.train_dataloader():
    #     enc_speech_cepstra, _, _, _, enc_reverb_speech_wav, _, rir, stochastic_noise, noise_condition, _, _ = batch
    #     print(enc_speech_cepstra.shape)

    # Example testing a dataset
    dataset = GTUIRDataset(cfg, type="train", split_train_val_test_p=[80,10,10], device='cuda')
    # iterate over the dataset
    for i in range(len(dataset)):
        audio_data, file_id = dataset[i]
  
if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("--config_path", type=str, default="config.yaml", 
        help="A full or relative path to a configuration yaml file. (default: config.yaml)")
        
    args = parser.parse_args()
    main(args)