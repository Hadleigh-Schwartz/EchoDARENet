"""
dEchorate IR dataset loader.
Code adapted from original EARS dataset code provided at https://github.com/sp-uhh/ears_benchmark/tree/main
"""

from torch.utils.data import Dataset, DataLoader
import numpy as np
import soundfile as sf
import glob
import os
import librosa
import sofa
import mat73


class dechorateIRDataset(Dataset):
    def __init__(self, config, type="train", split_train_val_test_p=[80, 10, 10], device='cuda'):
        self.config = config
        self.root_dir = os.path.join(os.path.expanduser(self.config['datasets_path']), 'ears_rirs')
        self.type = type
        
        if type == "train" and "dechorate" not in config.train_rir_datasets:
            print("WARNING: Loading dechorate dataset for training, but dechorate is not in the config's training rir datasets list. This may lead to unexpected results.")
        elif type == "val" and "dechorate" not in config.val_rir_datasets:
            print("WARNING: Loading dechorate dataset for validation, but dechorate is not in the config's validation rir datasets list. This may lead to unexpected results.")
        elif type == "test" and "dechorate" not in config.test_rir_datasets:
            print("WARNING: Loading dechorate dataset for testing, but dechorate is not in the config's testing rir datasets list. This may lead to unexpected results.")
        
        dir = os.path.join(self.root_dir, "dEchorate", "sofa")
        all_filenames = sorted(glob.glob(os.path.join(dir, "**", "*.sofa"), recursive=True))

        self.max_data_len = len(all_filenames)
        if type == "train" and "dechorate" not in config.val_rir_datasets and "dechorate" not in config.test_rir_datasets:
            # all idxs can be used for training, no need to split
            split_idxs = list(range(self.max_data_len))
        elif type == "val" and "dechorate"  not in config.train_rir_datasets and "dechorate"  not in config.test_rir_datasets:
            # all idxs can be used for validation, no need to split
            split_idxs = list(range(self.max_data_len))
        elif type == "test" and "dechorate"  not in config.train_rir_datasets and "dechorate"  not in config.val_rir_datasets:
            # all idxs can be used for testing, no need to split
            split_idxs = list(range(self.max_data_len))
        else:
            self.split_train_val_test_p = np.array(np.int16(split_train_val_test_p))
            self.split_train_val_test = np.int16(np.round( np.array(self.split_train_val_test_p)/100 * self.max_data_len ))
            self.split_edge = np.cumsum(np.concatenate(([0],self.split_train_val_test)), axis=0)
            self.idx_rand = np.random.RandomState(seed=config['random_seed']).permutation(self.max_data_len)
            split_idxs = []
            if self.type == "train":
                split_idxs  = self.idx_rand[self.split_edge[0]:self.split_edge[1]]
            elif self.type == "val":
                split_idxs = self.idx_rand[self.split_edge[1]:self.split_edge[2]]
            elif self.type == "test":
                split_idxs  = self.idx_rand[self.split_edge[2]:self.split_edge[3]]
        print(self.max_data_len, "total RIRs in dechorate dataset")
        self.split_filenames = [all_filenames[i] for i in split_idxs]
        print(len(self.split_filenames), "RIRs in split")
        self.samplerate = 48000
        exclude_filenames = self.get_exclude_rirs() # get excluded RIR filenames and remove them
        self.split_filenames = [f for f in self.split_filenames if f not in exclude_filenames]
        self.device = device

    def read_sofa_file(self, filename):
        hrtf = sofa.Database.open(filename)
        rir = hrtf.Data.IR.get_values()
        channel = np.random.randint(0, rir.shape[1])
        rir = rir[0,channel,:]
        sr_rir = hrtf.Data.SamplingRate.get_values().item()
        return rir, sr_rir

    def get_exclude_rirs(self):
        """
        Get list of filenames of RIRs that are excluded from the dataset based on their early reverb time

        Early reverb time is taken to coincide with the max value in the RIR
        """
        min_early_reverb = self.config["min_early_reverb"]
        exclude_rirs = []
        for idx in range(len(self.split_filenames)):
            filename = self.split_filenames[idx]
            rir, sr = self.read_sofa_file(filename)
            max_rir_idx = np.argmax(rir)
            max_rir_time = max_rir_idx / sr
            if max_rir_time < min_early_reverb:
                exclude_rirs.append(filename)
        return exclude_rirs

    def __len__(self):
        return len(self.split_filenames)

    def __getitem__(self, idx):
        path = self.split_filenames[idx]
        rir, sr = self.read_sofa_file(path)
        filename = os.path.basename(path)
        return rir, filename

 
def dechorateIRDataloader(config_path, type="train"):
    return DataLoader(dechorateIRDataset(config_path, type=type))