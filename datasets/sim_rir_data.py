from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import soundfile as sf
import glob
import os

class SimIRDataset(Dataset):
    def __init__(self, config, type="train", split_train_val_test_p=[80,10,10], device='cuda'):
        self.config = config
        self.root_dir = Path(os.path.expanduser(self.config['datasets_path']),'simulated_irs')
        self.type = type

        self.max_data_len = len(glob.glob(str(Path(self.root_dir,"*.wav"))))

        # get the sample rate from any file in the directory
        self.samplerate = sf.read(glob.glob(str(Path(self.root_dir,"*.wav")))[0])[1]

        self.split_train_val_test_p = np.array(np.int16(split_train_val_test_p))
        self.split_train_val_test = np.int16(np.round( np.array(self.split_train_val_test_p)/100 * self.max_data_len ))
        self.split_edge = np.cumsum(np.concatenate(([0],self.split_train_val_test)), axis=0)
        self.idx_rand = np.random.RandomState(seed=config['random_seed']).permutation(self.max_data_len)

        split = []
        if self.type == "train":
            split = self.idx_rand[self.split_edge[0]:self.split_edge[1]]
        elif self.type == "val":
            split = self.idx_rand[self.split_edge[1]:self.split_edge[2]]
        elif self.type == "test":
            split = self.idx_rand[self.split_edge[2]:self.split_edge[3]]

        files = glob.glob(str(Path(self.root_dir,"*.wav")))
        self.split_filenames = [files[i] for i in split]
    
        # get excluded RIR filenames and remove them
        exclude_filenames = self.get_exclude_rirs()

        self.split_filenames = [f for f in self.split_filenames if f not in exclude_filenames]
        self.device = device

    def get_exclude_rirs(self):
        """
        Get list of filenames of RIRs that are excluded from the dataset based on their early reverb time

        Early reverb time is taken to coincide with the max value in the RIR
        """
        min_early_reverb = self.config["min_early_reverb"]
        exclude_rirs = []
        # iterate through all RIRs 
        for idx in range(len(self.split_filenames)):
            filename = self.split_filenames[idx]
            audio_data, samplerate = sf.read(filename)
            max_rir_idx = np.argmax(audio_data)
            max_rir_time = max_rir_idx / samplerate
            if max_rir_time < min_early_reverb:
                exclude_rirs.append(filename)
        return exclude_rirs

    def __len__(self):
        return len(self.split_filenames)

    def __getitem__(self, idx):
        filename = self.split_filenames[idx]
        audio_data, samplerate = sf.read(filename)
        return audio_data, filename

 
def SimIRDataloader(config_path, type="train"):
    return DataLoader(SimIRDataset(config_path, type=type))