from torch.utils.data import Dataset
import soundfile as sf
import glob
import os
import numpy as np

class EARSSpeechDataset(Dataset):
    def __init__(self, config, type="train", device='cuda'):
        self.config = config
        self.root_dir = os.path.join(os.path.expanduser(self.config.datasets_path), f"ears")
        self.type = type

        # splits defined by EARS here: https://github.com/sp-uhh/ears_benchmark/blob/main/generate_ears_reverb.py#L227
        all_speakers = sorted(os.listdir(self.root_dir))
        valid_speakers = ["p100", "p101"] 
        test_speakers = ["p102", "p103", "p104", "p105", "p106", "p107"]
        
        speakers = {
            "train": [s for s in all_speakers if s not in valid_speakers + test_speakers],
            "valid": valid_speakers, 
            "test": test_speakers
        }

        speech_files = []
        for speaker in speakers[type]:  
            speech_files += sorted(glob.glob(os.path.join(self.root_dir, speaker, "*.wav")))  
        random_seed = self.config.random_seed
        np.random.seed(random_seed)
        np.random.shuffle(speech_files)
        self.speech_files = speech_files
        
        # determine samplerate by reading the first file
        first_file = speech_files[0]
        _, sr = sf.read(first_file)
        self.samplerate = sr
        assert self.samplerate == 48000, f"Samplerate of {first_file} is not 48000 Hz"

        self.num_files = len(speech_files)
    
    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        speech, sr = sf.read(self.speech_files[idx])
        return speech, sr

  
