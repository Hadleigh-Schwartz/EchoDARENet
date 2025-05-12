"""
Code adapted from original EARS dataset code provided at https://github.com/sp-uhh/ears_benchmark/tree/main
Note: Split ratios will be ignored as the dataset is already split into train, val, test
"""

from torch.utils.data import Dataset, DataLoader
import numpy as np
import soundfile as sf
import librosa
import glob
import os
import sofa
import mat73

class EARSIRDataset(Dataset):
    def __init__(self, config, type="train", split_train_val_test_p=[80,10,10], device='cuda'):
        self.config = config
        self.root_dir = os.path.join(os.path.expanduser(self.config['datasets_path']), 'ears_rirs')
        self.type = type
        
        # Splits for RIRs, defined by EARS here https://github.com/sp-uhh/ears_benchmark/blob/main/generate_ears_reverb.py#L119
        rir_files = {
            "train": [],
            "valid": [],
            "test": [],
        }

        # ACE-Challenge dataset
        dir = os.path.join(self.root_dir, "ACE-Challenge")
        names = ["Chromebook", "Crucif", "EM32", "Lin8Ch", "Mobile", "Single"]
        for name in names:
            rir_files["test"] += sorted(glob.glob(os.path.join(dir, name, "**", "*RIR.wav"), recursive=True))

        # AIR dataset
        dir = os.path.join(self.root_dir, "AIR", "AIR_1_4", "AIR_wav_files")
        rir_files["valid"] += sorted(glob.glob(os.path.join(dir, "*.wav")))

        # ARNI dataset
        dir = os.path.join(self.root_dir, "ARNI")
        all_arni_files = sorted(glob.glob(os.path.join(dir, "**", "*.wav"), recursive=True))
        # remove file numClosed_26-35/IR_numClosed_28_numComb_2743_mic_4_sweep_5.wav because it is corrupted
        all_arni_files = [file for file in all_arni_files if "numClosed_26-35/IR_numClosed_28_numComb_2743_mic_4_sweep_5.wav" not in file]
        rir_files["train"] += sorted(list(np.random.choice(all_arni_files, size=1000, replace=False))) # take 1000 of 132037 RIRs

        # BRUDEX dataset
        dir = os.path.join(self.root_dir, "BRUDEX")
        rir_files["train"] += sorted(glob.glob(os.path.join(dir, "rir", "**", "*.mat"), recursive=True))

        # dEchorate dataset
        dir = os.path.join(self.root_dir, "dEchorate", "sofa")
        rir_files["train"] += sorted(glob.glob(os.path.join(dir, "**", "*.sofa"), recursive=True))

        # DetmoldSRIR dataset
        dir = os.path.join(self.root_dir, "DetmoldSRIR")
        rir_files["train"] += sorted(glob.glob(os.path.join(dir, "SetA_SingleSources", "Data", "**", "*.wav"), recursive=True))

        # Palimpsest dataset
        dir = os.path.join(self.root_dir, "Palimpsest")
        rir_files["train"] += sorted(glob.glob(os.path.join(dir, "**", "*.wav"), recursive=True))
            

        self.split_filenames = rir_files[type]
        self.samplerate = 48000 # for EARS, have to upsample some 
        # get excluded RIR filenames and remove them
        exclude_filenames = self.get_exclude_rirs()


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
    return DataLoader(EARSIRDataset(config_path, type=type))