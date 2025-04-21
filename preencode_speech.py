from argparse import ArgumentParser
import numpy as np
import os
import sys 
import pickle
import soundfile as sf
import librosa


from datasets.speech_data import LibriSpeechDataset
from utils.utils import getConfig

# append echo encoding parent dir to path
curr_dir = os.getcwd()
echo_dir = curr_dir.split("EchoDARENet")[0] 
sys.path.append(echo_dir)
from traditional_echo_hiding import encode, decode, create_filter_bank

def main(args):
    cfg = getConfig(config_path=args.config_path)
    
    samplerate = cfg["sample_rate"]
    amplitude = cfg["Encoding"]["amplitude"]
    delays = cfg["Encoding"]["delays"]
    win_size = cfg["Encoding"]["win_size"]
    kernel = cfg["Encoding"]["kernel"]
    decoding = cfg["Encoding"]["decoding"]
    assert decoding in ["autocepstrum", "cepstrum"], "Invalid decoding method specified. Choose either 'autocepstrum' or 'cepstrum'."
    filters = create_filter_bank(kernel, delays, amplitude)

    cutoff_freq = cfg["Encoding"]["cutoff_freq"]
    nwins = cfg["nwins"]
    normalize = cfg["model"]["params"]["normalize"]
    noise_condition_length = cfg["model"]["params"]["noise_condition_length"]
    reverb_speech_duration = nwins * win_size

    # make the data directory if it does not exist
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    # copy the config file to the data directory
    with open(os.path.join(args.data_dir, "config.yaml"), "w") as f:
        with open(args.config_path, "r") as f2:
            f2_content = f2.read()
            f.write(f2_content)
              
    for split in ["train", "val", "test"]:
        if not os.path.exists(os.path.join(args.data_dir, split)):
            os.makedirs(os.path.join(args.data_dir, split))
        meta = {}
        speech_dataset = LibriSpeechDataset(cfg, type=split)
        if args.max_samples is None:
            max_samples = len(speech_dataset)
        else:
            max_samples = args.max_samples

        for i, data in enumerate(speech_dataset):
            if i >= max_samples:
                break
            speech = data[0].flatten().numpy()
            og_sr = data[1]

            # pad if not at least self.reverb_speech_duration
            speech = np.pad(
                speech,
                pad_width=(0, np.max((0, reverb_speech_duration - len(speech)))),
            )
            speech = librosa.resample(speech,
                orig_sr=og_sr,
                target_sr=samplerate,
                res_type='soxr_hq')  
            num_wins = int(speech.shape[0] / win_size)
            symbols = np.random.randint(0, len(delays), size = num_wins)
            speech = speech[:num_wins * win_size] # trim the speech to be a multiple of the window size. 
            enc_speech = encode(speech, symbols, amplitude, delays, win_size, samplerate, kernel, filters = filters)
            sf.write(os.path.join(args.data_dir, split, f"enc_{i}.wav"), enc_speech, samplerate)
            sf.write(os.path.join(args.data_dir, split, f"og_{i}.wav"), speech, samplerate)
            meta[i] = {
                "symbols": symbols
            }
   
        with open(os.path.join(args.data_dir, f"{split}_meta.pkl"), "wb") as f:
            pickle.dump(meta, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, default="cfg.yaml", 
                        help="A full or relative path to a cfguration yaml file. (default: cfg.yaml)")
    parser.add_argument("--data_dir", "-d", type=str, default=None, required=True, 
                        help = "A full or relative path to the data directory to store the encoded files at.")
    parser.add_argument("--max_samples", "-m", type=int, default=None,
                        help = "Maximum number of speech samples to encode. If None, all speech samples in the split will be encoded.")
    args = parser.parse_args()
    main(args)
