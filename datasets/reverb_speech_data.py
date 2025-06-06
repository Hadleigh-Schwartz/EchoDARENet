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
    
    def get_wavenet_item(self, idx):
        idx_speech = idx % len(self.speech_dataset)
        idx_rir    = idx % len(self.rir_dataset)
        speech = self.speech_dataset[idx_speech][0].flatten()
        speech = np.pad(
            speech,
            pad_width=(0, np.max((0,self.config["rir_duration"] - len(speech)))),
            )
        speech = speech[:self.config["rir_duration"]]
        if self.echo_encode:
            num_wins = len(speech) // self.win_size
            symbols = np.random.randint(0, len(self.delays), size = num_wins)
            speech = speech[:num_wins * self.win_size] # trim the speech to be a multiple of the window size. 
            speech = encode(speech, symbols, self.amplitude, self.delays, self.win_size, self.samplerate, self.kernel)

            # decode encoded speech to get baseline error rate
            pred_symbols, pred_symbols_autocepstrum  = decode(speech, self.delays, self.win_size, self.samplerate, pn = None, gt = symbols, plot = False, cutoff_freq = self.cutoff_freq)
            num_errs = np.sum(np.array(pred_symbols) != np.array(symbols))
            num_errs_autocepstrum  = np.sum(np.array(pred_symbols_autocepstrum ) != np.array(symbols))
            if self.decoding == "autocepstrum":
                num_errs_no_reverb = num_errs_autocepstrum 
            elif self.decoding == "cepstrum":
                num_errs_no_reverb = num_errs 
            else:
                raise Exception("Unknown decoding method. Specify 'autocepstrum' or 'cepstrum'.")
            # print(f"Encoded is {len(speech)} samples. Num wins is {num_wins}. Num errors symbols og: {num_errs}/{len(pred_symbols)}. Num errors symbols autocep {num_errs_autocepstrum }/{len(pred_symbols_autocepstrum )}"))
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

        if self.config["prealign_rir"]:
            # Fix the misaligment between the clean speech and the reverberant speech (possibly matters for wave-u-net)
            reverb_speech = reverb_speech[4096:] # was 4097, reverb speech seemed to be one sample early.
        
        # must get the reverb errs before padding
        if self.config["echo_encode"]:
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

            
        # plt.plot(speech, label = "orig", alpha = 0.5)
        # plt.plot(reverb_speech, label = "reverb", alpha = 0.5)
        # plt.legend()
        # plt.savefig(f"speech_samps/plots/{idx}.png")
        # plt.clf()

        if self.config["norm_audio"]:
            speech = speech - np.mean(speech)
            denom = np.max(np.abs(speech))
            if denom != 0:
                speech = speech / denom
            else:
                speech = np.zeros_like(speech)
            reverb_speech = reverb_speech - np.mean(reverb_speech)
            denom_reverb = np.max(np.abs(reverb_speech))
            if denom_reverb != 0:
                reverb_speech = reverb_speech / denom_reverb
            else:
                reverb_speech = np.zeros_like(reverb_speech)
         
        # fig, ax = plt.subplots(3, 1, figsize=(10, 6), tight_layout=True)
        # ax[0].plot(speech, label = "orig", alpha = 0.5)
        # ax[0].set_title("Original Speech")
        # ax[0].set_xlim(0, 32777)
        # ax[1].plot(reverb_speech, label = "reverb", alpha = 0.5)
        # ax[1].set_title("Reverberant Speech")
        # ax[1].set_xlim(0, 32777)
        # ax[2].plot(rir, label = "rir", alpha = 0.5)
        # ax[2].set_title("RIR")
        # ax[2].set_xlim(0, 2000)
        # plt.savefig(f"speech_samps/plots/{idx}.png")
        # plt.clf()

        reverb_speech = np.pad(
            reverb_speech,
            pad_width=(0, np.max((0,135141 - len(reverb_speech)))),
            )
        reverb_speech = reverb_speech[:135141] # expected input size given 15 up, 5 down filters and 2sec output

        speech = np.pad(
            speech,
            pad_width=(0, np.max((0,135141 - len(speech)))),
            )
        speech = speech[:135141]               # expected input size given 15 up, 5 down filters and 2sec output

        # if np.max(reverb_speech) > 8 and np.min(reverb_speech) < -5:
        #     print(idx, np.max(reverb_speech), np.min(reverb_speech))

        # # check if reverb speech has any nones or infinities
        # if np.any(np.isnan(reverb_speech)):
        #     print(f"NaN found in reverb speech at index {idx}")
        # if np.any(np.isinf(reverb_speech)):
        #     print(f"Inf found in reverb speech at index {idx}")

        return reverb_speech, speech, rir, symbols, num_errs_no_reverb, num_errs_reverb

    def get_dare_item(self, idx):
        """
        Original Dare datalaoder with echo encoding added
        """
        idx_speech = idx % len(self.speech_dataset)
        idx_rir    = idx % len(self.rir_dataset)

        speech = self.speech_dataset[idx_speech][0].flatten()

        speech = np.pad(
            speech,
            pad_width=(0, np.max((0,self.reverb_speech_duration - len(speech)))),
            )
        speech = speech[:self.reverb_speech_duration]
        if self.echo_encode:
            num_wins = len(speech) // self.win_size
            symbols = np.random.randint(0, len(self.delays), size = num_wins)
            speech = speech[:num_wins * self.win_size] # trim the speech to be a multiple of the window size. 
            speech = encode(speech, symbols, self.amplitude, self.delays, self.win_size, self.samplerate, self.kernel)

            # decode encoded speech to get baseline error rate
            pred_symbols, pred_symbols_autocepstrum  = decode(speech, self.delays, self.win_size, self.samplerate, pn = None, gt = symbols, plot = False, cutoff_freq = 1000)
            num_errs = np.sum(np.array(pred_symbols) != np.array(symbols))
            num_errs_autocepstrum  = np.sum(np.array(pred_symbols_autocepstrum ) != np.array(symbols))
            if self.decoding == "autocepstrum":
                num_errs_no_reverb = num_errs_autocepstrum
            elif self.decoding == "cepstrum":
                num_errs_no_reverb = num_errs 
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

        reverb_speech = np.pad(
            reverb_speech,
            pad_width=(0, np.max((0,self.reverb_speech_duration - len(reverb_speech)))),
            )
        reverb_speech = reverb_speech[:self.reverb_speech_duration]
        
        if self.config["echo_encode"]:
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
            
    
        reverb_speech_stft = librosa.stft(
            reverb_speech,
            n_fft=self.nfft,
            hop_length=self.nhop,
            win_length=self.nfft,
            window='hann'
        )

        if self.stft_format == 'magphase':
            np.seterr(divide = 'ignore')
            rs_mag = np.log(np.abs(reverb_speech_stft)) # Magnitude
            np.seterr(divide = 'warn')
            rs_mag[np.isinf(rs_mag)] = self.eps
            # Normalize to [-1,1]
            rs_mag = rs_mag - rs_mag.min()
            rs_mag = rs_mag / rs_mag.max() / 2 - 1

            reverb_speech = np.stack((rs_mag, np.angle(reverb_speech_stft)))
        elif self.stft_format == 'realimag':
            reverb_speech = np.stack((np.real(reverb_speech_stft), np.imag(reverb_speech_stft)))
            reverb_speech = reverb_speech - np.mean(reverb_speech)
            reverb_speech = reverb_speech / np.max(np.abs(reverb_speech))
        else:
            raise Exception("Unknown STFT format. Specify 'realimag' or 'magphase'.")

        speech_wav = speech / np.max(np.abs(speech)) - np.mean(speech)
        speech_stft = librosa.stft(
            speech,
            n_fft=self.nfft,
            hop_length=self.nhop,
            win_length=self.nfft,
            window='hann'
            )

        if self.stft_format_sp == 'magphase':
            np.seterr(divide = 'ignore')
            s_mag = np.log(np.abs(speech_stft)) # Magnitude
            np.seterr(divide = 'warn')
            s_mag[np.isinf(s_mag)] = self.eps
            # Normalize to [-1,1]
            s_mag = s_mag - s_mag.min()
            s_mag = s_mag / s_mag.max() / 2 - 1
            speech = np.stack((s_mag, np.angle(speech_stft)))

        elif self.stft_format_sp == 'realimag':
            speech = np.stack((np.real(speech_stft), np.imag(speech_stft)))
            speech = speech - np.mean(speech)
            speech = speech / np.max(np.abs(speech))

        else:
            raise Exception("Unknown STFT format. Specify 'realimag' or 'magphase'.")
        rir_fft = np.fft.rfft(rir)
        rir_fft = np.stack((np.real(rir_fft), np.imag(rir_fft)))
        rir_fft = rir_fft - np.mean(rir_fft)
        rir_fft = rir_fft / np.max(np.abs(rir_fft))

        # trim speech_wav by one sample to match the inverse STFT output in the model 
        return reverb_speech, speech, speech_wav, rir_fft[:,:,None], rir, rirfn, symbols, num_errs_no_reverb, num_errs_reverb

    def __getitem__(self, idx):
        if not self.data_in_ram or (self.data_in_ram and self.idx_to_data[idx] == -1):
            if self.model == 'Waveunet':
                reverb_speech, speech, rir, symbols, num_errs_no_reverb, num_errs_reverb = self.get_wavenet_item(idx)
            else:
                reverb_speech, speech, speech_wav, rir_fft, rir, rirfn, symbols, num_errs_no_reverb, num_errs_reverb = self.get_dare_item(idx)
            
            if self.data_in_ram:
                if self.model == 'Waveunet':
                    self.data.append((reverb_speech, speech, rir, symbols, num_errs_no_reverb, num_errs_reverb))
                else:
                    self.data.append((reverb_speech, speech, speech_wav, rir_fft, rir, rirfn, symbols, num_errs_no_reverb, num_errs_reverb))
                self.idx_to_data[idx] = len(self.data) - 1
            
        else:
            if self.model == 'Waveunet':
                reverb_speech, speech, rir, symbols, num_errs_no_reverb, num_errs_reverb = self.data[self.idx_to_data[idx]]
               
            else:
                reverb_speech, speech, speech_wav, rir_fft, rir, rirfn, symbols, num_errs_no_reverb, num_errs_reverb = self.data[self.idx_to_data[idx]]
               
        if self.model == "Waveunet":
            # make symbols to tensor on speech device
            return reverb_speech, speech, rir, symbols, num_errs_no_reverb, num_errs_reverb
        else:  
            return reverb_speech, speech, speech_wav, rir_fft, rir, rirfn, symbols, num_errs_no_reverb, num_errs_reverb
        
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
