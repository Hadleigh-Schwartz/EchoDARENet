from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from pytorch_lightning import LightningDataModule
from datasets.speech_data import LibriSpeechDataset
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

# append echo encoding parent dir to path
curr_dir = os.getcwd()
echo_dir = curr_dir.split("EchoDARENet")[0] 
sys.path.append(echo_dir)
from traditional_echo_hiding import encode, decode

class DareDataset(Dataset):
    def __init__(self, config, type="train", split_train_val_test_p=[80,10,10], device='cuda'):

        self.type = type
        self.split_train_val_test_p = split_train_val_test_p
        self.device = device

        self.config = config

        # wavenet specific stuff
        self.model  = config['Model']['model_name']
        self.waveunet_input_samples  = 73721
        self.waveunet_output_samples = 32777
        ###################
        
        
        self.rir_dataset = MitIrSurveyDataset(self.config, type=self.type, device=device)
        self.speech_dataset = LibriSpeechDataset(self.config, type=self.type)
        
        self.dataset_len = self.config['DataLoader']['batch_size'] * self.config['Trainer']["limit_"+type+"_batches"]
        
        self.stft_format = self.config['stft_format']
        self.stft_format_sp = self.config['stft_format_sp']
        self.eps = 10**-32

        self.nfft = self.config['nfft']
        self.nfrms = self.config['nfrms']
        self.samplerate = self.speech_dataset[0][1]

        self.rir_duration = config['rir_duration']
        self.rir_sos = signal.butter(6, 40, 'hp', fs=self.samplerate, output='sos') # Seems that this is largely for denoising the RIRs??

        self.nhop = self.config['nhop']
        self.reverb_speech_duration = self.nfrms * self.nhop

        self.data_in_ram = config['data_in_ram']
        if self.data_in_ram:
            self.data = []
            self.idx_to_data = -np.ones((len(self.speech_dataset) * len(self.rir_dataset),),dtype=np.int32) 

        
        if config["echo_encode"]:
            np.random.seed(config['random_seed']) # for echo_encoding symbol generation
            self.echo_encode = True
            self.amplitude = config["Encoding"]["amplitude"]
            self.delays = config["Encoding"]["delays"]
            self.win_size = config["Encoding"]["win_size"]
            self.kernel = config["Encoding"]["kernel"]
            self.decoding = config["Encoding"]["decoding"]
            assert self.decoding in ["autocepstrum", "cepstrum"], "Invalid decoding method specified. Choose either 'autocepstrum' or 'cepstrum'."
            self.cutoff_freq = config["Encoding"]["cutoff_freq"]
        else:
            self.echo_encode = False

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
        # print("b:", b)
        # print("a:", a)
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
        

    def get_dare_item(self, idx):
        """
        Original Dare datalaoder with echo encoding added
        """
        idx_speech = idx % len(self.speech_dataset)
        idx_rir    = idx % len(self.rir_dataset)

        speech = self.speech_dataset[idx_speech][0].flatten()

        # pad if not at least self.reverb_speech_duration
        speech = np.pad(
            speech,
            pad_width=(0, np.max((0,self.reverb_speech_duration - len(speech)))),
            )

        if self.echo_encode:
            num_wins = len(speech) // self.win_size
            symbols = np.random.randint(0, len(self.delays), size = num_wins)
            speech = speech[:num_wins * self.win_size] # trim the speech to be a multiple of the window size. 
            speech = encode(speech, symbols, self.amplitude, self.delays, self.win_size, self.samplerate, self.kernel)
        else:
            symbols = 0
            num_errs_no_reverb = 0

        rir,rirfn = self.rir_dataset[idx_rir]
        rir = rir.flatten()
        rir = rir[~np.isnan(rir)]

        rir = librosa.resample(rir,
            orig_sr=self.rir_dataset.samplerate,
            target_sr=self.samplerate,
            res_type='soxr_hq')
        rir = rir - np.mean(rir)
        rir = rir / np.max(np.abs(rir))
        maxI = np.argmax(np.abs(rir))

        rir = rir[25:]
        rir = rir * signal.windows.tukey(rir.shape[0], alpha=2*25/rir.shape[0], sym=True) # Taper 50 samples at the beginning and end of the RIR
        rir = signal.sosfilt(self.rir_sos, rir) # not sure we want this??
        maxI = np.argmax(np.abs(rir))
        rir = rir / rir[maxI] # scaling
        rir = np.pad(
            rir,
            pad_width=(0, np.max((0,self.rir_duration - len(rir)))), # 0-padding at end has no effect on reverb results so is ok
            )
        if self.config["prealign_rir"]:
            rir = np.concatenate((np.zeros(4096-maxI),rir[:-4096+maxI])) # i assume this is the pre-aligmnent step. I'm deleting it because it throws off decoding..

        reverb_speech = signal.convolve(speech, rir, method='fft', mode = "full")

        # randomly select a :self.reverb_speech_duration portion of speech,
        start_options = [i for i in range(0, len(speech) - self.reverb_speech_duration + self.win_size, self.win_size)]
        start = np.random.choice(start_options)
        speech = speech[start:start + self.reverb_speech_duration]
        reverb_speech = reverb_speech[start:start + self.reverb_speech_duration]
        start_win = start // self.win_size
        symbols = symbols[start_win:start_win + self.reverb_speech_duration // self.win_size]
        # reverb_speech = np.pad(
        #     reverb_speech,
        #     pad_width=(0, np.max((0,self.reverb_speech_duration - len(reverb_speech)))),
        #     )
        # reverb_speech = reverb_speech[]
        
        if self.config["echo_encode"]:
            # decode encoded speech to get baseline error rate
            pred_symbols, pred_symbols_autocepstrum  = decode(speech, self.delays, self.win_size, self.samplerate, pn = None, gt = symbols, plot = False, cutoff_freq = 1000)
            num_errs = np.sum(np.array(pred_symbols) != np.array(symbols))
            num_errs_autocepstrum  = np.sum(np.array(pred_symbols_autocepstrum ) != np.array(symbols))
            if self.decoding == "autocepstrum":
                num_errs_no_reverb = num_errs_autocepstrum
            elif self.decoding == "cepstrum":
                num_errs_no_reverb = num_errs 
            # print(f"Num err symbols og: {num_errs}/{len(pred_symbols)}. Num err symbols autocep {num_errs_autocepstrum }/{len(pred_symbols_autocepstrum )}")

            reverb_speech_dec = reverb_speech[:num_wins * self.win_size]
            rev_pred_symbols, rev_pred_symbols_autocepstrum  = decode(reverb_speech_dec, self.delays, self.win_size, self.samplerate, pn = None, gt = symbols, plot = False, cutoff_freq = 1000)
            rev_num_errs = np.sum(np.array(rev_pred_symbols) != np.array(symbols))
            rev_num_errs_autocepstrum  = np.sum(np.array(rev_pred_symbols_autocepstrum ) != np.array(symbols))
            # print(f"Reverbed num err symbols og: {rev_num_errs}/{len(rev_pred_symbols)}. Reverbed num err symbols autocep {rev_num_errs_autocepstrum }/{len(rev_pred_symbols_autocepstrum )}")
            if self.decoding == "autocepstrum":
                num_errs_reverb = rev_num_errs_autocepstrum
            elif self.decoding == "cepstrum":
                num_errs_reverb = rev_num_errs
        else:
            num_errs_reverb = 0
        
        reverb_speech_cepstra = self.get_cepstrum_windows(reverb_speech)
        reverb_speech_cepstra = np.dstack(reverb_speech_cepstra)
    
        speech_wav = speech / np.max(np.abs(speech)) - np.mean(speech)
     
        # get cepstrum windows of speech
        speech_cepstra = self.get_cepstrum_windows(speech_wav)
        speech_cepstra = np.dstack(speech_cepstra)

        rir_fft = np.fft.rfft(rir)
        rir_fft = np.stack((np.real(rir_fft), np.imag(rir_fft)))
        rir_fft = rir_fft - np.mean(rir_fft)
        rir_fft = rir_fft / np.max(np.abs(rir_fft))

        # trim speech_wav by one sample to match the inverse STFT output in the model 
        return reverb_speech_cepstra, speech_cepstra, speech_wav, rir_fft[:,:,None], rir, rirfn, symbols, num_errs_no_reverb, num_errs_reverb

    def __getitem__(self, idx):
        """
        TODO: move get_dare_item stuff to here
        """
        if not self.data_in_ram or (self.data_in_ram and self.idx_to_data[idx] == -1):
           
            reverb_speech_cepstra, speech_cepstra, speech_wav, rir_fft, rir, rirfn, symbols, num_errs_no_reverb, num_errs_reverb = self.get_dare_item(idx)
            
            if self.data_in_ram:
                self.data.append((reverb_speech_cepstra, speech_cepstra, speech_wav, rir_fft, rir, rirfn, symbols, num_errs_no_reverb, num_errs_reverb))
                self.idx_to_data[idx] = len(self.data) - 1
            
        else:
            reverb_speech_cepstra, speech_cepstra, speech_wav, rir_fft, rir, rirfn, symbols, num_errs_no_reverb, num_errs_reverb = self.data[self.idx_to_data[idx]]
        
        return reverb_speech_cepstra, speech_cepstra, speech_wav, rir_fft, rir, rirfn, symbols, num_errs_no_reverb, num_errs_reverb
        
def DareDataloader(config,type="train"):
    cfg = copy.deepcopy(config)
    if type != "train":
        cfg['DataLoader']['shuffle'] = False
    return DataLoader(DareDataset(cfg,type),**cfg['DataLoader'])

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
