from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import soundfile as sf
import glob
import os

class SoundCamIRDataset(Dataset):
    """
    The SoundCam IR dataset has IRs from 10 microphones per room+human pair, for anywhere from 100-1000 configurations of humans
    within each room. Because variations induced by different human positions/identities are very minor, using all IRs per room would 
    be redundant. Instead, we randomly select one configuration (a human identity and position) per room, but do use all 10 microphones,
    as mic (and its associated different position within the room) has a bigger impact on IR.
    """
    def __init__(self, config, type="train", split_train_val_test_p=[50, 25, 25], device='cuda'):
        self.config = config
        self.root_dir = Path(os.path.expanduser(self.config['datasets_path']), 'SoundCamFlat')
        self.type = type
  
        self.samplerate = 48000 # hardcoded (based on paper) because data is saved as .npy files, which don't have a samplerate specified

        room_folders = glob.glob(f"{self.root_dir}/*")
        if type == "train" and "SoundCam" not in config.val_rir_datasets and "SoundCam" not in config.test_rir_datasets:
            # all rooms can be used for training, no need to split
            split_room_idxs = [0, 1, 2, 3]
        elif type == "val" and "SoundCam" not in config.train_rir_datasets and "SoundCam" not in config.test_rir_datasets:
            # all rooms can be used for validation, no need to split
            split_room_idxs = [0, 1, 2, 3]
        elif type == "test" and "SoundCam" not in config.train_rir_datasets and "SoundCam" not in config.val_rir_datasets:
            # all rooms can be used for testing, no need to split
            split_room_idxs = [0, 1, 2, 3]
        else:
            # get all rooms and randomly assign them to train, val, test, ensuring no room is shared between splits
            # (would potentially cause leakage if it occurred)
            if self.type == "train":
                split_room_idxs = [0, 1]
            elif self.type == "val":
                split_room_idxs = [2]
            elif self.type == "test":
                split_room_idxs = [3]
            print(len(split_room_idxs), "rooms in split")
        
        # for each room, randomly choose a human folder, corresponding to a specific person within the room
        # then, within that human folder, randomly choose a configuration, corresponding to the human's position within it
        self.split_filenames_and_mics = []
        for room_idx in split_room_idxs:
            room_folder = room_folders[room_idx]
            human_folders = glob.glob(f"{room_folder}/*")
            human_idx = np.random.randint(0, len(human_folders)) # randomly chosen human identity
            human_folder = human_folders[human_idx]
            num_configs = len(glob.glob(f"{human_folder}/*"))
            config_num = np.random.randint(0, num_configs) # randomly chosen human position, i.e., room config
            for mic_num in range(10): # append the filename and configuration number 10 times, one for each mic
                # get the filename
                filename = f"{human_folder}/config{config_num}_deconvolved.npy"
                # append the filename and mic number
                self.split_filenames_and_mics.append((filename, mic_num))

        exclude_case_ids = self.get_exclude_rirs() # get excluded RIR filenames and remove them

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