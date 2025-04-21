from torch.utils.data import Dataset
from utils.utils import getConfig
from pathlib import Path
import soundfile as sf
import glob
import os
import pickle

class EncodedLibriSpeechDataset(Dataset):
    def __init__(self, config, type="train", device='cuda'):
        
        self.config = config
        self.root_dir = Path(os.path.expanduser(self.config["datasets_path"]), f"preencoded_librispeech/")
        self.data_dir = f"{self.root_dir}/{type}"
        enc_config = getConfig(config_path=f"{self.root_dir}/config.yaml")
        assert self.config == enc_config, "Config mismatch. The config used for generation is different from the config used for training."

        self.type = type

        self.num_files = len(glob.glob(f"{self.data_dir}/enc_*.wav"))
        with open(os.path.join(self.root_dir, f"{type}_meta.pkl"), "rb") as f:
            self.meta = pickle.load(f)
   
    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        enc_filename = os.path.join(self.data_dir, f"enc_{idx}.wav")
        og_filename = os.path.join(self.data_dir, f"og_{idx}.wav")
        enc_speech, samplerate = sf.read(enc_filename)
        og_speech, _ = sf.read(og_filename)
        symbols = self.meta[idx]["symbols"]
        return og_speech, enc_speech, samplerate, symbols

  
