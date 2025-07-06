from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import soundfile as sf
import glob
import os
import requests
import zipfile
from tqdm import tqdm
import shutil

class MITIRSurveyDataset(Dataset):
    def __init__(self, config, type="train", split_train_val_test_p=[80,10,10], device='cuda', download=True):
        self.config = config
        self.root_dir = Path(os.path.expanduser(self.config['datasets_path']),'MIT_IR_Survey')
        self.type = type

        if type == "train" and "MIT" not in config.train_rir_datasets:
            print("WARNING: Loading MIT dataset for training, but MIT is not in the config's training rir datasets list. This may lead to unexpected results.")
        elif type == "val" and "MIT" not in config.val_rir_datasets:
            print("WARNING: Loading MIT dataset for validation, but MIT is not in the config's validation rir datasets list. This may lead to unexpected results.")
        elif type == "test" and "MIT" not in config.test_rir_datasets:
            print("WARNING: Loading MIT dataset for testing, but MIT is not in the config's testing rir datasets list. This may lead to unexpected results.")
        
        
        if download and not os.path.isdir(str(self.root_dir)): # If the path doesn't exist, download the dataset if set to true
            self.download_mit_ir_survey(self.root_dir)

        self.max_data_len = 270 # This is supposed to be 271 but there is an IR missing in the dataset
        self.samplerate = 32000


        if type == "train" and "MIT" not in config.val_rir_datasets and "MIT" not in config.test_rir_datasets:
            # all idxs can be used for training, no need to split
            split_idxs = list(range(self.max_data_len))
        elif type == "val" and "MIT"  not in config.train_rir_datasets and "MIT"  not in config.test_rir_datasets:
            # all idxs can be used for validation, no need to split
            split_idxs = list(range(self.max_data_len))
        elif type == "test" and "MIT"  not in config.train_rir_datasets and "MIT"  not in config.val_rir_datasets:
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
        print(self.max_data_len, "total RIRs in MIT IR Survey dataset")
        files = glob.glob(str(Path(self.root_dir,"*")))
        self.split_filenames = [files[i] for i in split_idxs]
        print(len(self.split_filenames), "RIRs in split")
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
        if samplerate != self.samplerate:
            raise Exception("The samplerate of the audio in the dataset is not 32kHz.")
        filename = os.path.basename(filename)  # Get just the filename without the path for this part
        return audio_data, filename

    def download_mit_ir_survey(self, local_path):
        url = "https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip"
        local_path = os.path.expanduser(local_path)

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(str(local_path)+".zip", 'wb') as f:
                for chunk in tqdm(r.iter_content(chunk_size=8192), total=1424, desc="Downloading MIT IR Survey"): 
                    f.write(chunk)

        with zipfile.ZipFile(str(local_path)+".zip", 'r') as zip_ref:
            zip_ref.extractall(str(local_path))

        files_list = os.listdir(str(Path(local_path,"Audio")))
        for file in files_list:
            if file != ".DS_Store":
                os.rename(str(Path(local_path,"Audio",file)), str(Path(local_path,file)))
        
        for dir in ([d[0] for d in os.walk(str(local_path))])[1:]:
            if os.path.isdir(dir):
                shutil.rmtree(dir)
        
        print("Download complete.")

        return True

def MITIRSurveyDataloader(config_path, type="train"):
    return DataLoader(MITIRSurveyDataset(config_path, type=type))