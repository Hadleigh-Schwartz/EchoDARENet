from torch.utils.data import Dataset
from utils.utils import load_config
from pathlib import Path
import soundfile as sf
import glob
import os
import pickle

class EncodedSpeechDataset(Dataset):
    def __init__(self, config,  preencoded_speech_path, type="train", device='cuda'):
        self.config = config
        self.root_dir = Path(os.path.expanduser(self.config.datasets_path), preencoded_speech_path)
        self.data_dir = f"{self.root_dir}/{type}"
        enc_config = load_config(f"{self.root_dir}/config.yaml")
        # assert self.config.Encoding == enc_config.Encoding, f"Encoding configuration mismatch. The encoding parameters used for creation of encoded speech stored at {self.root_dir} is different from that of the parameters in the config being passed here."

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
        og_speech, og_sr = sf.read(og_filename)
        # sanity check
        assert og_sr == samplerate, f"Sample rate mismatch. The sample rate of the original speech {og_sr} is different from that of the encoded speech {samplerate}."
        symbols = self.meta[idx]["symbols"]
        return og_speech, enc_speech, samplerate, symbols

  
