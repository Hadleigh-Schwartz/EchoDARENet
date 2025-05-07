from argparse import ArgumentParser
import numpy as np
import os
import sys 
import pickle
import soundfile as sf
import librosa
from pathlib import Path
import random
import torch as t
import pandas as pd

from datasets.librispeech_data import LibriSpeechDataset
from datasets.hifi_speech_data import HiFiSpeechDataset
from utils.utils import load_config

# append echo encoding parent dir to path
curr_dir = os.getcwd()
echo_dir = curr_dir.split("EchoDARENet")[0] 
sys.path.append(echo_dir)
from traditional_echo_hiding import encode, create_filter_bank

def main(args):
    cfg = load_config(args.config_path)

    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    t.manual_seed(cfg.random_seed)
    
    samplerate = cfg.sample_rate
    amplitude = cfg.Encoding.amplitude
    delays = cfg.Encoding.delays
    win_size = cfg.Encoding.win_size
    kernel = cfg.Encoding.kernel
    hanning_factor = cfg.Encoding.hanning_factor
    decoding = cfg.Encoding.decoding
    assert decoding in ["autocepstrum", "cepstrum"], "Invalid decoding method specified. Choose either 'autocepstrum' or 'cepstrum'."
    filters = create_filter_bank(kernel, delays, amplitude)

    nwins = cfg.nwins
    reverb_speech_duration = nwins * win_size

    # make the data directory if it does not exist
    data_dir = Path(os.path.expanduser(cfg.datasets_path), cfg.preencoded_speech_path)
    data_dir = str(data_dir)


    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # copy the config file to the data directory
    with open(os.path.join(data_dir, "config.yaml"), "w") as f:
        with open(args.config_path, "r") as f2:
            f2_content = f2.read()
            f.write(f2_content)
              
    for split in ["train", "val", "test"]:
        if not os.path.exists(os.path.join(data_dir, split)):
            os.makedirs(os.path.join(data_dir, split))
        meta = {}
        
        if cfg.speech_dataset == "HiFi":
            speech_dataset = HiFiSpeechDataset(cfg, type=split)
        elif cfg.speech_dataset == "LibriSpeech":
            speech_dataset = LibriSpeechDataset(cfg, type=split)
        else:
            raise ValueError(f"Selected speech dataset {cfg.speech_dataset} is not valid")
        
        if args.max_files is None:
            max_files = len(speech_dataset)
        else:
            max_files = args.max_files

        num_files_written = 0
        for i, data in enumerate(speech_dataset):
            # skip if the file already exists
            if os.path.exists(os.path.join(data_dir, split, f"enc_{num_files_written}.wav")):
                num_files_written += 1
                continue

            if num_files_written >= max_files:
                break
            if cfg.speech_dataset == "LibriSpeech":
                speech = data[0].flatten().numpy() # librispeech not flattened
            else:
                speech = data[0]
            og_sr = data[1]

            # resample to common samplerate specified in config
            speech = librosa.resample(speech,
                orig_sr=og_sr,
                target_sr=samplerate,
                res_type='soxr_hq')  
            
            # crop speech to remove silence
            if args.no_silence:
                speech_pos = speech + -1*np.min(speech)
                df = pd.Series(speech_pos)
                df = df.rolling(100, center=True).median()
                df = df + np.min(speech)
                first_index = df[df > 0.001].index[0]
                last_index = df[df > 0.001].index[-1]
                speech = speech[first_index:last_index]
                if len(speech) < reverb_speech_duration:
                    continue
            else:  
                # pad if not at least self.reverb_speech_duration
                speech = np.pad(
                    speech,
                    pad_width=(0, np.max((0, reverb_speech_duration - len(speech)))),
                )

            num_wins = int(speech.shape[0] / win_size)
            symbols = np.random.randint(0, len(delays), size = num_wins)
            speech = speech[:num_wins * win_size] # trim the speech to be a multiple of the window size. 
            enc_speech = encode(speech, symbols, amplitude, delays, win_size, samplerate, kernel, filters = filters, hanning_factor = hanning_factor)
            sf.write(os.path.join(data_dir, split, f"enc_{num_files_written}.wav"), enc_speech, samplerate)
            sf.write(os.path.join(data_dir, split, f"og_{num_files_written}.wav"), speech, samplerate)
            meta[num_files_written] = {
                "symbols": symbols
            }

            num_files_written += 1
   
        with open(os.path.join(data_dir, f"{split}_meta.pkl"), "wb") as f:
            pickle.dump(meta, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, 
                        help="A full or relative path to a cfguration yaml file. (default: cfg.yaml)")
    parser.add_argument("--max_files", "-m", type=int, default=None,
                        help = "Maximum number of speech files to encode. If None, all speech files in the split will be encoded.")
    parser.add_argument("--no_silence", action="store_true",
                        help = "If set, the speech will be cropped to remove silence and only files with non-silent content of at least reverb_duration will be used." \
                        "If not set, the speech may be padded to the required length.")
    args = parser.parse_args()
    main(args)
