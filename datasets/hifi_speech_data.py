from torch.utils.data import Dataset
from utils.utils import load_config
from pathlib import Path
import soundfile as sf
import glob
import os
import pickle
import numpy as np

class HiFiSpeechDataset(Dataset):
    def __init__(self, config, type="train", device='cuda'):
        self.config = config
        self.root_dir = Path(os.path.expanduser(self.config.datasets_path), f"hi_fi_tts_v0/audio")
        self.type = type

        split_ratio = [0.8, 0.1, 0.1] # 80% train, 10% val, 10% test

        # get all hifi audio files and shuffly
        all_audio_files = glob.glob(f"{self.root_dir}/*/*/*.flac")
        random_seed = self.config.random_seed
        np.random.seed(random_seed)
        np.random.shuffle(all_audio_files)

        # determine samplerate by reading the first file
        first_file = all_audio_files[0]
        _, sr = sf.read(first_file)
        self.samplerate = sr

        # split the audio files into train, val, and test sets
        train_files = all_audio_files[:int(len(all_audio_files) * split_ratio[0])]
        val_files = all_audio_files[int(len(all_audio_files) * split_ratio[0]):int(len(all_audio_files) * (split_ratio[0] + split_ratio[1]))]
        test_files = all_audio_files[int(len(all_audio_files) * (split_ratio[0] + split_ratio[1])):]
        if type == "train":
            self.files = train_files
        elif type == "val":
            self.files = val_files
        elif type == "test":
            self.files = test_files

        self.num_files = len(self.files)
    
    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        speech, sr = sf.read(self.files[idx])
        return speech, sr

  
