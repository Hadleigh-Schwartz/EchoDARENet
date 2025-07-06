"""
Code adapted from original EARS dataset code provided at https://github.com/sp-uhh/ears_benchmark/tree/main
"""

from torch.utils.data import Dataset, DataLoader
import numpy as np
import soundfile as sf
import librosa
import glob
import os
import sofa
import mat73



class ACEIRDataset(Dataset):
    def __init__(self, config, type="train", split_train_val_test_p=[80, 10, 10], device='cuda'):
        self.config = config
        self.root_dir = os.path.join(os.path.expanduser(self.config['datasets_path']), 'ears_rirs')
        self.type = type
        

        # ACE-Challenge dataset
        all_filenames = []
        dir = os.path.join(self.root_dir, "ACE-Challenge")
        names = ["ACE_Corpus_RIRN_Chromebook/Chromebook", "ACE_Corpus_RIRN_Crucif/Crucif", "ACE_Corpus_RIRN_EM32/EM32", 
                 "ACE_Corpus_RIRN_Lin8Ch/Lin8Ch", "ACE_Corpus_RIRN_Mobile/Mobile", "ACE_Corpus_RIRN_Single/Single"]
        for name in names:
            all_filenames += sorted(glob.glob(os.path.join(dir, name, "**", "*RIR.wav"), recursive=True))

        self.max_data_len = len(all_filenames)
        if type == "train" and "ACE" not in config.val_rir_datasets and "ACE" not in config.test_rir_datasets:
            # all idxs can be used for training, no need to split
            split_idxs = list(range(self.max_data_len))
        elif type == "val" and "ACE"  not in config.train_rir_datasets and "ACE"  not in config.test_rir_datasets:
            # all idxs can be used for validation, no need to split
            split_idxs = list(range(self.max_data_len))
        elif type == "test" and "ACE"  not in config.train_rir_datasets and "ACE"  not in config.val_rir_datasets:
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


        self.split_filenames = [all_filenames[i] for i in split_idxs]
        self.samplerate = 48000 # for EARS, have to upsample some 
        exclude_filenames = self.get_exclude_rirs() # get excluded RIR filenames and remove them
        self.split_filenames = [f for f in self.split_filenames if f not in exclude_filenames]
        self.device = device


    def read_ears_rir_file(self, rir_file):
        """
        Read EARS RIR from file based on extension and upsample, using code from original authors
        https://github.com/sp-uhh/ears_benchmark/blob/main/generate_ears_reverb.py#L119 for consistency
        """
        if "ARNI" in rir_file:
            rir, sr_rir = sf.read(rir_file, always_2d=True)
            # Take random channel if file is multi-channel
            channel = np.random.randint(0, rir.shape[1])
            rir = rir[:,channel]
            assert sr_rir == 44100, "What"
            rir = librosa.resample(rir, orig_sr=sr_rir, target_sr=48000)
            sr_rir = 48000
        elif rir_file.endswith(".wav"):
            rir, sr_rir = sf.read(rir_file, always_2d=True)
            # Take random channel if file is multi-channel
            channel = np.random.randint(0, rir.shape[1])
            rir = rir[:,channel]
        elif rir_file.endswith(".sofa"):
            hrtf = sofa.Database.open(rir_file)
            rir = hrtf.Data.IR.get_values()
            channel = np.random.randint(0, rir.shape[1])
            rir = rir[0,channel,:]
            sr_rir = hrtf.Data.SamplingRate.get_values().item()
        elif rir_file.endswith(".mat"):
            rir = mat73.loadmat(rir_file)
            sr_rir = rir["fs"].item()
            rir = rir["data"]
            channel = np.random.randint(0, rir.shape[1])
            rir = rir[:,channel]
        else:
            raise ValueError(f"Unknown file format: {rir_file}")

        assert sr_rir == self.samplerate, f"Sampling rate of {rir_file} is {sr_rir} Hz and not {self.samplerate} Hz"
        assert rir.ndim == 1, f"RIR {rir_file} is not 1D, but {rir.ndim}D"

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
            rir, sr = self.read_ears_rir_file(filename)
            max_rir_idx = np.argmax(rir)
            max_rir_time = max_rir_idx / sr
            if max_rir_time < min_early_reverb:
                exclude_rirs.append(filename)
        return exclude_rirs

    def __len__(self):
        return len(self.split_filenames)

    def __getitem__(self, idx):
        path = self.split_filenames[idx]
        audio_data, sr = self.read_ears_rir_file(path)
        filename = os.path.basename(path)
        return audio_data, filename

 
def HomulaIRDataloader(config_path, type="train"):
    return DataLoader(ACEIRDataset(config_path, type=type))