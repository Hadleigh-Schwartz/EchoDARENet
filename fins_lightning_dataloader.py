from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from pytorch_lightning import LightningDataModule

from datasets.speech_data import LibriSpeechDataset
from datasets.preencoded_speech_data import EncodedLibriSpeechDataset
from datasets.rir_data import MitIrSurveyDataset
import librosa
from scipy import signal
import numpy as np
import torch as t
import copy
import os
import sys
import soundfile as sf
import matplotlib.pyplot as plt
import torch

# append echo encoding parent dir to path
curr_dir = os.getcwd()
echo_dir = curr_dir.split("EchoDARENet")[0] 
sys.path.append(echo_dir)
from traditional_echo_hiding import encode, decode, create_filter_bank

class DareDataset(Dataset):
    def __init__(self, config, type="train", split_train_val_test_p=[80,10,10], device='cuda'):

        self.type = type
        self.split_train_val_test_p = split_train_val_test_p
        self.device = device

        self.config = config        
        self.preencoded_speech = config["preencoded_speech"]
        self.rir_dataset = MitIrSurveyDataset(self.config, type=self.type, device=device)

        if self.preencoded_speech:
            self.speech_dataset = EncodedLibriSpeechDataset(self.config, type=self.type)
        else:    
            self.speech_dataset = LibriSpeechDataset(self.config, type=self.type)
        
        self.dataset_len = self.config['DataLoader']['batch_size'] * self.config['Trainer']["limit_"+type+"_batches"]
        
        self.samplerate = self.config['sample_rate']
        self.rir_length = int(config["dataset"]["params"]["rir_duration"] * config["dataset"]["params"]["sr"])  # 1 sec = 48000 samples
      
        self.data_in_ram = config['data_in_ram']
        if self.data_in_ram:
            self.data = []
            self.idx_to_data = -np.ones((len(self.speech_dataset) * len(self.rir_dataset),),dtype=np.int32) 

        # np.random.seed(config['random_seed']) # for echo_encoding symbol generation already set in main though
        self.amplitude = config["Encoding"]["amplitude"]
        self.delays = config["Encoding"]["delays"]
        self.win_size = config["Encoding"]["win_size"]
        self.kernel = config["Encoding"]["kernel"]
        self.decoding = config["Encoding"]["decoding"]
        assert self.decoding in ["autocepstrum", "cepstrum"], "Invalid decoding method specified. Choose either 'autocepstrum' or 'cepstrum'."
        self.filters = create_filter_bank(self.kernel, self.delays, self.amplitude)

        self.cutoff_freq = config["Encoding"]["cutoff_freq"]
        self.nwins = config["nwins"]
        self.normalize = config["model"]["params"]["normalize"]
        self.noise_condition_length = config["model"]["params"]["noise_condition_length"]
        self.reverb_speech_duration = self.nwins * self.win_size
     
        
    
    def __len__(self):
        return self.dataset_len
    
    def butter_highpass(self, cutoff, fs, order=5):
        """
        Create a Butterworth high-pass filter.

        Parameters:
            cutoff : float
                The cutoff frequency for the high-pass filter.
            fs : int
                The sampling rate of the signal.
            order : int
                The order of the Butterworth filter.
        
        Returns:
            tuple : The filter coefficients.
        """
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def butter_highpass_filter(self, data, cutoff, fs, order=5):
        """
        Apply a Butterworth high-pass filter to a signal.

        Parameters: 
            data : np.ndarray
                The signal to filter.
            cutoff : float
                The cutoff frequency for the high-pass filter.
            fs : int
                The sampling rate of the signal.
            order : int
                The order of the Butterworth filter.
        
        Returns:
            np.ndarray : The filtered signal.
        """
        b, a = self.butter_highpass(cutoff, fs, order=order)
        y = signal.filtfilt(b, a, data)
        return y
    
    def get_cepstrum_windows(self, audio):
        num_wins = len(audio) // self.win_size
    
        if self.cutoff_freq is not None:
            audio = self.butter_highpass_filter(audio, self.cutoff_freq, self.samplerate)

        win_cepstra = []
        for i in range(num_wins):
            win = audio[i * self.win_size : (i + 1) * self.win_size]

            fft = np.fft.fft(win)
            sqr_log_fft = np.log(np.abs(fft) + 0.00001)
            cepstrum = np.fft.ifft(sqr_log_fft)
            # only need the first half because second half is just the conjugate of the first half
            cepstrum = cepstrum[:len(cepstrum) // 2]
            win_cepstra.append(np.stack((np.real(cepstrum), np.imag(cepstrum))))
        return win_cepstra      
    
    def rms_normalize(self, sig, rms_level=0.1):
        """
        RMS normalize as optionally done in original FINS implementation.
        (https://github.com/kyungyunlee/fins/blob/main/fins/utils/audio.py)

        Parameters:
            sig : np.ndarray (signal_length)
                The signal to normalize.
            rms_level : float
                Linear gain value for the RMS normalization.
        
        Returns:
            np.ndarray : The normalized signal (signal_length)

        """
        # linear rms level and scaling factor
        # r = 10 ** (rms_level / 10.0)
        a = np.sqrt((sig.shape[0] * rms_level**2) / (np.sum(sig**2) + 1e-7))

        # normalize
        y = sig * a
        return y
    
    def peak_normalize(self, sig, peak_val):
        """
        Normalize the signal to a peak value as optionally done in original FINS implementation.
        (https://github.com/kyungyunlee/fins/blob/main/fins/utils/audio.py)

        Parameters:
            sig : np.ndarray (channel, signal_length)
                The signal to normalize.
            peak_val : float
        
        Returns:
            np.ndarray : The normalized signal (channel, signal_length)
        """
        peak = np.max(np.abs(sig[:, :512]), axis=-1, keepdims=True)
        sig = np.divide(sig, peak + 1e-7)
        sig = sig * peak_val
        return sig
        
    def __getitem__(self, idx):
        if not self.data_in_ram or (self.data_in_ram and self.idx_to_data[idx] == -1):
            """
            Original Dare datalaoder with echo encoding added
            """
            idx_speech = idx % len(self.speech_dataset)
            idx_rir    = idx % len(self.rir_dataset)
            
            if self.preencoded_speech:
                # encoded librispeech dataset
                speech, enc_speech, _, symbols = self.speech_dataset[idx_speech]
                num_wins = len(enc_speech) // self.win_size
            else:
                # normal librispeech dataset
                speech = self.speech_dataset[idx_speech][0].flatten()
                # pad if not at least self.reverb_speech_duration
                speech = np.pad(
                    speech,
                    pad_width=(0, np.max((0,self.reverb_speech_duration - len(speech)))),
                )
                speech = librosa.resample(speech,
                    orig_sr=16000,
                    target_sr=self.samplerate,
                    res_type='soxr_hq')
                num_wins = len(speech) // self.win_size
                symbols = np.random.randint(0, len(self.delays), size = num_wins)
                speech = speech[:num_wins * self.win_size] # trim the speech to be a multiple of the window size. 
                enc_speech = encode(speech, symbols, self.amplitude, self.delays, self.win_size, self.samplerate, self.kernel, filters = self.filters)

            # TODO: remove DC components as in o.g., FINS implementation? Seems not realistic to do this prior to convolution

            rir, rirfn = self.rir_dataset[idx_rir]
            rir = rir.flatten()
            rir = rir[~np.isnan(rir)]
            rir = librosa.resample(rir,
                orig_sr=self.rir_dataset.samplerate,
                target_sr=self.samplerate,
                res_type='soxr_hq')
            rir /= np.max(np.abs(rir)) * 0.999
            if len(rir) < self.rir_length:
                rir = np.pad(rir, (0, self.rir_length - len(rir)))
            elif len(rir) > self.rir_length:
                rir = rir[:self.rir_length]
            
            # convolve
            enc_reverb_speech = signal.convolve(enc_speech, rir, method='fft', mode = "full")
            unenc_reverb_speech = signal.convolve(speech, rir, method='fft', mode = "full")

            # randomly select a :self.reverb_speech_duration portion of speech,
            start_options = [i for i in range(0, len(enc_speech) - self.reverb_speech_duration + self.win_size, self.win_size)]
            start = np.random.choice(start_options)
            enc_speech = enc_speech[start:start + self.reverb_speech_duration]
            enc_reverb_speech = enc_reverb_speech[start:start + self.reverb_speech_duration]
            unenc_reverb_speech = unenc_reverb_speech[start:start + self.reverb_speech_duration]
            start_win = start // self.win_size
            symbols = symbols[start_win:start_win + self.reverb_speech_duration // self.win_size]

            # get cepstrum windows of speech (encoded, unencoded, reverberated and whatnot)
            enc_speech_cepstra = self.get_cepstrum_windows(enc_speech)
            enc_speech_cepstra = np.dstack(enc_speech_cepstra)
            enc_reverb_speech_cepstra = self.get_cepstrum_windows(enc_reverb_speech)
            enc_reverb_speech_cepstra = np.dstack(enc_reverb_speech_cepstra)
            unenc_reverb_speech_cepstra = self.get_cepstrum_windows(unenc_reverb_speech)
            unenc_reverb_speech_cepstra = np.dstack(unenc_reverb_speech_cepstra)

            # normalize the time domain speech samples.
            # caution: this normalization is not reflected in the cepstra and this should be considered when 
            # training with any losses using time domain representations
            # enc_speech_wav = enc_speech / np.max(np.abs(enc_speech)) - np.mean(enc_speech)
            if self.normalize == "peak":
                # peak normalization from original FINS implementation
                raise NotImplementedError("Peak normalization not implemented yet.")
            elif self.normalize == "rms":
                # RMS normalization from original FINS implementation
                enc_speech_wav = self.rms_normalize(enc_speech, rms_level = self.config["model"]["params"]["rms_level"])
                enc_reverb_speech_wav = self.rms_normalize(enc_reverb_speech, rms_level = self.config["model"]["params"]["rms_level"])
                unenc_reverb_speech_wav = self.rms_normalize(unenc_reverb_speech, rms_level = self.config["model"]["params"]["rms_level"])
            else:
                enc_speech_wav = enc_speech
                enc_reverb_speech_wav = enc_reverb_speech
                unenc_reverb_speech_wav = unenc_reverb_speech

            # unsqueeze wavs to add channel dimension (expected by FINS)
            enc_speech_wav = np.expand_dims(enc_speech_wav, axis=0)
            enc_reverb_speech_wav = np.expand_dims(enc_reverb_speech_wav, axis=0)
            unenc_reverb_speech_wav = np.expand_dims(unenc_reverb_speech_wav, axis=0)

            stochastic_noise = torch.randn((1, self.rir_length))
            stochastic_noise = stochastic_noise.repeat(self.config["model"]["params"]["num_filters"], 1)
            noise_condition = torch.randn((self.config["model"]["params"]["noise_condition_length"]))

            if self.data_in_ram:
                self.data.append((enc_speech_cepstra, enc_reverb_speech_cepstra, unenc_reverb_speech_cepstra, 
                                    enc_speech_wav, enc_reverb_speech_wav, unenc_reverb_speech_wav,
                                    rir, stochastic_noise, noise_condition, symbols, idx_rir))
                self.idx_to_data[idx] = len(self.data) - 1
            
        else:
            enc_speech_cepstra, enc_reverb_speech_cepstra, unenc_reverb_speech_cepstra, 
            enc_speech_wav, enc_reverb_speech_wav, unenc_reverb_speech_wav,
            rir, stochastic_noise, noise_condition, symbols, idx_rir = self.data[self.idx_to_data[idx]]
        
        return enc_speech_cepstra, enc_reverb_speech_cepstra, unenc_reverb_speech_cepstra, \
                enc_speech_wav, enc_reverb_speech_wav, unenc_reverb_speech_wav, \
                rir, stochastic_noise, noise_condition, symbols,  idx_rir


def batch_sampler(config, type="train"):
    dummy_dataset = DareDataset(config, type=type)
    dataset_len = len(dummy_dataset)
    batch_size = config['DataLoader']['batch_size']
    # dividing the dataset into batches
    num_batches = dataset_len // batch_size
    indices = np.arange(dataset_len)
    if type != "train":
        # use batches of increasing nums, not randomized  
        indices = np.array_split(indices, num_batches)
    else:
        # emulate a vanilla random sampler, i.e., shuffle the indices, then divide into batches
        np.random.shuffle(indices)
        indices = np.array_split(indices, num_batches)
    return indices    

def DareDataloader(config,type="train"):
    cfg = copy.deepcopy(config)
    # if type != "train":
    #     cfg['DataLoader']['shuffle'] = False
    return DataLoader(DareDataset(cfg,type), num_workers = cfg["DataLoader"]["num_workers"], persistent_workers = cfg["DataLoader"]["persistent_workers"], pin_memory = cfg["DataLoader"]["pin_memory"], 
                       batch_sampler=batch_sampler(cfg, type))

class DareDataModule(LightningDataModule):
    def __init__(self,config):
        super().__init__()
        self.config = config
    def train_dataloader(self):
        return DareDataloader(type="train",config=self.config)
    def val_dataloader(self):
        return DareDataloader(type="val",config=self.config)
    def test_dataloader(self):
        return DareDataloader(type="test",config=self.config)