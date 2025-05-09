from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import soundfile as sf
import glob
import os

class SoundCamIRDataset(Dataset):
    def __init__(self, config, type="train", split_train_val_test_p=[80,10,10], device='cuda'):
        self.config = config
        self.root_dir = Path(os.path.expanduser(self.config['datasets_path']), 'SoundCamFlat')
        self.type = type
  
        # get the sample rate from any file in the directory
        self.samplerate = 48000

        room_folders = glob.glob(f"{self.root_dir}/*")
        # randomly assign room folders to train, val, and test
        self.splittrain_val_test_p = np.array(np.int16(split_train_val_test_p))
        self.splittrain_val_test_p = self.splittrain_val_test_p / np.sum(self.splittrain_val_test_p) * 100
        self.splittrain_val_test_p = np.int16(np.round(self.splittrain_val_test_p))
        self.split_edge = np.cumsum(np.concatenate(([0],self.splittrain_val_test_p)), axis=0)
        self.idx_rand = np.random.RandomState(seed=config['random_seed']).permutation(len(room_folders))

        # self.split_train_val_test = np.int16(np.round( np.array(self.split_train_val_test_p)/100 * self.max_data_len ))
        # self.split_edge = np.cumsum(np.concatenate(([0],self.split_train_val_test)), axis=0)
        # self.idx_rand = np.random.RandomState(seed=config['random_seed']).permutation(self.max_data_len)

        if self.type == "train":
            split_room_idxs = self.idx_rand[self.split_edge[0]:self.split_edge[1]]
        elif self.type == "val":
            split_room_idxs = self.idx_rand[self.split_edge[1]:self.split_edge[2]]
        elif self.type == "test":
            split_room_idxs = self.idx_rand[self.split_edge[2]:self.split_edge[3]]
    
        # for each room, randomly choose a number of humans and a configuration file. 
        self.split_filenames_and_mics = []
        for room_idx in split_room_idxs:
            room_folder = room_folders[room_idx]
            # get the number of human cases
            human_folders = glob.glob(f"{room_folder}/*")
            # randomly choose a human case
            human_idx = np.random.randint(0, len(human_folders))
            # get the human folder
            human_folder = human_folders[human_idx]
            num_configs = len(glob.glob(f"{human_folder}/*"))
            # randomly choose a configuration
            config_num = np.random.randint(0, num_configs)  
            # append the filename and configuration number 10 times, one for each mic
            for mic_num in range(10):
                # get the filename
                filename = f"{human_folder}/config{config_num}_deconvolved.npy"
                # append the filename and mic number
                self.split_filenames_and_mics.append((filename, mic_num))

        print(f"Number of {self.type} files: {len(self.split_filenames_and_mics)}")
        # get excluded RIR filenames and remove them
        exclude_case_ids = self.get_exclude_rirs()

        self.split_filenames_and_mics = [self.split_filenames_and_mics[i] for i in range(len(self.split_filenames_and_mics)) if i not in exclude_case_ids]
        self.device = device

    def get_exclude_rirs(self):
        """
        Get list of filenames of RIRs that are excluded from the dataset based on their early reverb time

        Early reverb time is taken to coincide with the max value in the RIR
        """
        min_early_reverb = self.config["min_early_reverb"]
        exclude_rirs = []
        # iterate through all RIRs 
        for i in range(len(self.split_filenames_and_mics)):
            filename = self.split_filenames_and_mics[i][0]
            mic_num = self.split_filenames_and_mics[i][1]
            all_rirs_data = np.load(filename)
            audio_data = all_rirs_data[mic_num, :]
            max_rir_idx = np.argmax(audio_data)
            max_rir_time = max_rir_idx / self.samplerate
            if max_rir_time < min_early_reverb:
                exclude_rirs.append(i)
        return exclude_rirs

    def __len__(self):
        return len(self.split_filenames_and_mics)

    def __getitem__(self, idx):
        file_path = self.split_filenames_and_mics[idx][0]
        filename = os.path.basename(file_path)
        mic_num = self.split_filenames_and_mics[idx][1]
        audio_data = np.load(file_path)[mic_num, :]
        return audio_data, filename

def SoundCamIRDataloader(config_path, type="train"):
    return DataLoader(SoundCamIRDataset(config_path, type=type))