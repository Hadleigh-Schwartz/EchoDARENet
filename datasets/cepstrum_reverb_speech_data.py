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
        
        self.rir_dataset = MitIrSurveyDataset(self.config, type=self.type, device=device)
        self.speech_dataset = LibriSpeechDataset(self.config, type=self.type)
        
        self.dataset_len = self.config['DataLoader']['batch_size'] * self.config['Trainer']["limit_"+type+"_batches"]
        
        self.samplerate = self.config['sample_rate']
        assert self.samplerate == self.speech_dataset[0][1] 
     
        self.same_batch_rir = self.config['same_batch_rir']
        if self.same_batch_rir:
            self.batch_size = self.config['DataLoader']['batch_size'] # need to know this so we can fix the RIR indices per batch using only idx
        self.rir_duration = config['rir_duration']
        self.rir_sos = signal.butter(6, 40, 'hp', fs=self.samplerate, output='sos') # Seems that this is largely for denoising the RIRs??

        self.data_in_ram = config['data_in_ram']
        if self.data_in_ram:
            self.data = []
            self.idx_to_data = -np.ones((len(self.speech_dataset) * len(self.rir_dataset),),dtype=np.int32) 

        # np.random.seed(config['random_seed']) # for echo_encoding symbol generation already set in main though
        self.amplitude = config["Encoding"]["amplitude"]
        self.delays = config["Encoding"]["delays"]
        self.win_size = config["Encoding"]["win_size"]
        self.kernel = config["Encoding"]["kernel"]
        self.hanning_factor = config["Encoding"]["hanning_factor"]
        self.decoding = config["Encoding"]["decoding"]
        assert self.decoding in ["autocepstrum", "cepstrum"], "Invalid decoding method specified. Choose either 'autocepstrum' or 'cepstrum'."
        self.cutoff_freq = config["Encoding"]["cutoff_freq"]
        self.nwins = config["nwins"]
        
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
        

    def get_dare_item(self, idx):
        """
        Original Dare datalaoder with echo encoding added
        """
        idx_speech = idx % len(self.speech_dataset)
        if self.same_batch_rir:
            idx_rir = (idx // self.batch_size) % len(self.rir_dataset)
        else:
            idx_rir    = idx % len(self.rir_dataset)
               
        speech = self.speech_dataset[idx_speech][0].flatten()

        # pad if not at least self.reverb_speech_duration
        speech = np.pad(
            speech,
            pad_width=(0, np.max((0,self.reverb_speech_duration - len(speech)))),
            )
        num_wins = len(speech) // self.win_size
        symbols = np.random.randint(0, len(self.delays), size = num_wins)
        speech = speech[:num_wins * self.win_size] # trim the speech to be a multiple of the window size. 
        enc_speech = encode(speech, symbols, self.amplitude, self.delays, self.win_size, self.samplerate, self.kernel, hannign_factor = self.hanning_factor)
    
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
        # reverb_speech = np.pad(
        #     reverb_speech,
        #     pad_width=(0, np.max((0,self.reverb_speech_duration - len(reverb_speech)))),
        #     )
        # reverb_speech = reverb_speech[]
        

        # decode encoded speech to get baseline error rate
        pred_symbols, pred_symbols_autocepstrum  = decode(enc_speech, self.delays, self.win_size, self.samplerate, pn = None, gt = symbols, plot = False, cutoff_freq = 1000)
        num_errs = np.sum(np.array(pred_symbols) != np.array(symbols))
        num_errs_autocepstrum  = np.sum(np.array(pred_symbols_autocepstrum ) != np.array(symbols))
        if self.decoding == "autocepstrum":
            num_errs_no_reverb = num_errs_autocepstrum
        elif self.decoding == "cepstrum":
            num_errs_no_reverb = num_errs 
        # print(f"Num err symbols og: {num_errs}/{len(pred_symbols)}. Num err symbols autocep {num_errs_autocepstrum }/{len(pred_symbols_autocepstrum )}")

        reverb_speech_dec = enc_reverb_speech[:num_wins * self.win_size]
        rev_pred_symbols, rev_pred_symbols_autocepstrum  = decode(reverb_speech_dec, self.delays, self.win_size, self.samplerate, pn = None, gt = symbols, plot = False, cutoff_freq = 1000)
        rev_num_errs = np.sum(np.array(rev_pred_symbols) != np.array(symbols))
        rev_num_errs_autocepstrum  = np.sum(np.array(rev_pred_symbols_autocepstrum ) != np.array(symbols))
        # print(f"Reverbed num err symbols og: {rev_num_errs}/{len(rev_pred_symbols)}. Reverbed num err symbols autocep {rev_num_errs_autocepstrum }/{len(rev_pred_symbols_autocepstrum )}")
        if self.decoding == "autocepstrum":
            num_errs_reverb = rev_num_errs_autocepstrum
        elif self.decoding == "cepstrum":
            num_errs_reverb = rev_num_errs

        # get cepstrum windows of speech
        enc_speech_cepstra = self.get_cepstrum_windows(enc_speech)
        enc_speech_cepstra = np.dstack(enc_speech_cepstra)

        enc_reverb_speech_cepstra = self.get_cepstrum_windows(enc_reverb_speech)
        enc_reverb_speech_cepstra = np.dstack(enc_reverb_speech_cepstra)

        unenc_reverb_speech_cepstra = self.get_cepstrum_windows(unenc_reverb_speech)
        unenc_reverb_speech_cepstra = np.dstack(unenc_reverb_speech_cepstra)
    
        enc_speech_wav = enc_speech / np.max(np.abs(enc_speech)) - np.mean(enc_speech)
     
  
        rir_fft = np.fft.rfft(rir)
        rir_fft = np.stack((np.real(rir_fft), np.imag(rir_fft)))
        rir_fft = rir_fft - np.mean(rir_fft)
        rir_fft = rir_fft / np.max(np.abs(rir_fft))

        # trim speech_wav by one sample to match the inverse STFT output in the model 
        return enc_reverb_speech_cepstra, enc_speech_cepstra, unenc_reverb_speech_cepstra, enc_speech_wav, rir_fft[:,:,None], rir, rirfn, symbols, num_errs_no_reverb, num_errs_reverb, idx_rir

    def __getitem__(self, idx):
        """
        TODO: move get_dare_item stuff to here
        """
        if not self.data_in_ram or (self.data_in_ram and self.idx_to_data[idx] == -1):
           
            enc_reverb_speech_cepstra, enc_speech_cepstra, unenc_reverb_speech_cepstra, enc_speech_wav, rir_fft, rir, rirfn, symbols, num_errs_no_reverb, num_errs_reverb, idx_rir = self.get_dare_item(idx)
            
            if self.data_in_ram:
                self.data.append((enc_reverb_speech_cepstra, enc_speech_cepstra, unenc_reverb_speech_cepstra, enc_speech_wav, rir_fft, rir, rirfn, symbols, num_errs_no_reverb, num_errs_reverb, idx_rir))
                self.idx_to_data[idx] = len(self.data) - 1
            
        else:
            enc_reverb_speech_cepstra, enc_speech_cepstra, unenc_reverb_speech_cepstra, enc_speech_wav, rir_fft, rir, rirfn, symbols, num_errs_no_reverb, num_errs_reverb, idx_rir = self.data[self.idx_to_data[idx]]
        
        return enc_reverb_speech_cepstra, enc_speech_cepstra, unenc_reverb_speech_cepstra, enc_speech_wav, rir_fft, rir, rirfn, symbols, num_errs_no_reverb, num_errs_reverb, idx_rir


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
    elif config["same_batch_rir"]:
        # ensure each batch consists of the same RIR by having consecutive indices within batch
        batch_start_ids = np.arange(0, num_batches)
        np.random.shuffle(batch_start_ids)
        indices = []
        for i in batch_start_ids:
            indices.append(list(np.arange(i * batch_size, (i + 1) * batch_size)))
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