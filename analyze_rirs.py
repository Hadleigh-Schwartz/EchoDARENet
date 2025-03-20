"""
Analyze RIRs in dataset to determine the typical timing of early reverbs, to inform
encoding
"""

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from pytorch_lightning import LightningDataModule
from datasets.speech_data import LibriSpeechDataset
from datasets.rir_data import MitIrSurveyDataset
import librosa
from scipy import signal
import numpy as np
import torch as t
import copy
from utils.utils import getConfig
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

config_path = "configs/config.yaml"
cfg = getConfig(config_path=config_path)

early_reverb_idxs = []
exclude_train_filenames = []
exclude_test_filenames = []
exclude_val_filenames = []
early_cutoff = 130
for type in ["train", "test", "val"]:
    rir_dataset = MitIrSurveyDataset(cfg, type=type, device="cpu")
    # iterate through the rir dataset
    for i in range(len(rir_dataset)):
        rir, rirfn = rir_dataset[i]
        # get rolling max of rir
        env = pd.Series(rir).rolling(window=20, min_periods = 0).max()
        env = env.rolling(window=200, min_periods = 0).mean()

        # # find tallest peak in env
        # env_peaks, peak_info = find_peaks(env, width = (None, None), height = (None, None), plateau_size=(None, None))
        # peak_heights = peak_info["peak_heights"]
        # tallest_peak = np.argmax(peak_heights)
        # tallest_peak_left_edge = peak_info["left_edges"][tallest_peak]
        # tallest_peak_left_ip = peak_info["left_ips"][tallest_peak]
        # tallest_peak_left_base = peak_info["left_bases"][tallest_peak]

        # get index of rir
        max_rir_idx = np.argmax(rir)
        early_reverb_idxs.append(max_rir_idx)

        if max_rir_idx < early_cutoff:
            if type == "train":
                exclude_train_filenames.append(rirfn)
            elif type == "test":
                exclude_test_filenames.append(rirfn)
            elif type == "val":
                exclude_val_filenames.append(rirfn)
        
        # # plot the rir
        # plt.plot(rir)
        # plt.plot(env)
        # plt.vlines(max_rir_idx, 0, 1, colors='r')
        # plt.xlim(0, 1000)
        # plt.title(rirfn)
        # plt.savefig(f"rir_plots/rir_{type}_{i}.png")
        # plt.clf()

# print min, median, max of early reverb indexes
early_reverb_idxs = np.array(early_reverb_idxs)
print(f"Min early reverb index: {early_reverb_idxs.min()}")
print(f"Median early reverb index: {np.median(early_reverb_idxs)}")
print(f"Max early reverb index: {early_reverb_idxs.max()}")
# plot histogram of early reverb indexes
plt.hist(early_reverb_idxs, bins=100)
plt.title("Early Reverb Indexes")
# set xtics every 10
plt.xticks(np.arange(0, 1000, 50))
plt.savefig("early_reverb_histogram.png")

print("Exclude train")
print(exclude_train_filenames)
print("Exclude test")
print(exclude_test_filenames)
print("Exclude val")
print(exclude_val_filenames)