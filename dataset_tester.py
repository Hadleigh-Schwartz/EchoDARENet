"""
For sanity checking dataloaders/datasets.
Replicate the data loading logic of the main training script and make sure I can iterate through batches
and that batch elements look as expected.
"""

from argparse import ArgumentParser
from fins_lightning_dataloader import DareDataModule
from datasets.gtu_rir_data import GTUIRDataset
from datasets.soundcam_rir_data import SoundCamIRDataset
from datasets.ears_speech_data import EARSSpeechDataset
from datasets.ears_rir_data import EARSIRDataset

import torch as t
import matplotlib.pyplot as plt
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

    # Example testing Data Module
    # datamodule = DareDataModule(config=cfg)
    # # iterate over the dataloader
    # for batch in datamodule.train_dataloader():
    #     enc_speech_cepstra, enc_reverb_speech_cepstra, unenc_reverb_speech_cepstra, \
    #             enc_speech_wav, enc_reverb_speech_wav, unenc_reverb_speech_wav, \
    #             rir, stochastic_noise, noise_condition, symbols,  idx_rir, num_errs_no_reverb, num_errs_reverb = batch
    #     print(rir.shape)

    # Example testing an IR dataset
    # dataset = GTUIRDataset(cfg, type="train", split_train_val_test_p=[80,10,10], device='cuda')
    # dataset = SoundCamIRDataset(cfg, type="train", split_train_val_test_p=[80,10,10], device='cuda')
    dataset = EARSIRDataset(cfg, type="train", split_train_val_test_p=[80,10,10], device='cuda')
    # iterate over the dataset
    for i in range(len(dataset)):
        rir_data, file_id = dataset[i]
        # plot the rir and save it
        plt.plot(rir_data)
        plt.title(f"RIR {file_id}")
        plt.xlim(0, 10000)
        plt.savefig(f"rir_{file_id}.png")
        plt.close()
        break

    # Example testing a speech dataset
    # dataset = EARSSpeechDataset(cfg, type="train", device='cuda')
    # # iterate over the dataset
    # for i in range(len(dataset)):
    #     speech_data, sr = dataset[i]
    #     plt.plot(speech_data)
    #     plt.title(f"Speech {i}")
    #     plt.savefig(f"speech_{i}.png")
    #     break
  
if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("--config_path", type=str, default="config.yaml", 
        help="A full or relative path to a configuration yaml file. (default: config.yaml)")
        
    args = parser.parse_args()
    main(args)