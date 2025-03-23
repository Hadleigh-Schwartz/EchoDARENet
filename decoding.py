import torch
#import torch.nn as nn
from torch import optim, nn
import pytorch_lightning as pl
from torchaudio.functional import highpass_biquad

from WaveUnet.crop import centre_crop
from WaveUnet.resample import Resample1d
from WaveUnet.conv import ConvLayer

import auraloss # for MR-STFT loss 
import matplotlib.pyplot as plt # for diagnostics only

import numpy as np


class DecodingLoss(nn.Module):
    def __init__(self, delays, win_size, decoding, cutoff_freq, sample_rate, softargmax_beta=1e10):
        super(DecodingLoss, self).__init__()
        self.delays = delays
        self.win_size = win_size
        self.decoding = decoding
        self.cutoff_freq = cutoff_freq
        self.sample_rate = sample_rate
        self.softargmax_beta = softargmax_beta
    
    def efficient_forward(self, audio_batch, symbols_batch, num_errs_no_reverb_batch, num_errs_reverb_batch):
     
        num_wins = audio_batch.shape[2] // self.win_size
        max_audio_len = num_wins * self.win_size

        # chop up audio into audio_batch.shape[2] // win_size windows
        audio_batch = audio_batch.squeeze(1) # (batch_size, 1, num_samples) -> (batch_size, num_samples)
        audio_batch = audio_batch[:, :max_audio_len] # preemptively cut so all splis will be equal
        res = torch.tensor_split(audio_batch, num_wins, dim=1) # (batch_size, num_samples) -> tuple of  num_wins tensors of shape (batch_size, win_size)
        audio_batch = torch.stack(res, dim=1) # tuple-> (batch_size, num_wins, win_size)

        # make one (batch_size * num_wins, win_size) i.e., (audio_batch.shape[0] * (audio_batch.shape[2] // win_size windows), win_size), tensor  X of audio windows
        all_windows = audio_batch.reshape(audio_batch.shape[0] * audio_batch.shape[1], self.win_size) # (batch_size, num_wins, win_size) -> (batch_size * num_wins, win_size)

        # have function that computes the symbol for each window
        # call torch.vmap on X to get a tensor Y of shape (batch_size * num_wins, 1) of symbols
        # reshape Y into (batch_size, num_wins)
        # print(audio_batch.shape)

    def forward(self, audio_batch, symbols_batch, num_errs_no_reverb_batch, num_errs_reverb_batch):
        """
        TODO: check out these to make more efficient
        https://discuss.pytorch.org/t/apply-a-function-similar-to-map-on-a-tensor/51088/5
        https://pytorch.org/functorch/stable/generated/functorch.vmap.html

        Parameters:
            audio_batch : torch.Tensor (batch_size, 1, num_samples)
                Batch of audio samples in time domain
            symbols_batch : torch.Tensor (batch_size, num_symbols) 
                Batch of groudn-truth symbols that were encoded onto the clean speech
            num_errs_no_reverb_batch : torch.Tensor (batch_size)
                Batch of number of errors when decoding the encoded clean speech (these can occur due to confounding peaks in speech 
                cepstra and are independent of the reverb or network)
            num_errs_reverb_batch : torch.Tensor (batch_size)
                Batch of number of errors when decoding the encoded reverb speech 
        """

        # testing
        self.efficient_forward(audio_batch, symbols_batch, num_errs_no_reverb_batch, num_errs_reverb_batch)


        tot_symbol_errs = 0 # total symbol errors (differentiable)
        total_err_reduction = 0 # accrue total error reduction (differentiable)
        gt_symbol_err_rate_reverb = 0
        gt_symbol_err_rate_no_reverb = 0

        num_wins = audio_batch.shape[2] // self.win_size
        max_audio_len = num_wins * self.win_size
        for audio_idx in range(audio_batch.shape[0]):
            audio = audio_batch[audio_idx, 0, :max_audio_len]
            symbols = symbols_batch[audio_idx]
            num_errs_no_reverb = num_errs_no_reverb_batch[audio_idx]
            num_errs_reverb = num_errs_reverb_batch[audio_idx]
           
            curr_audio_num_err_symbols = 0
            pred_symbols = []

            if self.cutoff_freq is not None: # high-pass filter the window 
                audio = highpass_biquad(audio,  self.cutoff_freq , self.sample_rate) # FILTERING IS WRONG???
            for i in range(num_wins):
                win = audio[i * self.win_size: (i + 1) * self.win_size]

                if self.decoding == "cepstrum":
                    cepstrum = self.torch_get_cepstrum(win)
                    cep_vals = cepstrum[self.delays]
                    max_val = self.softargmax(cep_vals)
                    cep_loss_val = torch.clamp(torch.abs(max_val - symbols[i]), min = 0, max = 1)
                    curr_audio_num_err_symbols= curr_audio_num_err_symbols + cep_loss_val
                    pred_symbols.append(int(max_val.item()))

                    # # get actual decoded symbol (non-differentiable, for debug)
                    # cep_vals_nondiff = cep_vals.clone()
                    # cep_vals_nondiff = cep_vals_nondiff.detach().cpu().numpy()
                    # max_val_nondiff = np.argmax(cep_vals_nondiff)
                    # if max_val_nondiff != symbols[i]:
                    #     tot_symbol_errors += 1 
                else:
                    autocepstrum = self.torch_get_autocepstrum(win)
                    autocepstrum_vals  = autocepstrum[self.delays]
                    max_autocepstrum_val  = self.softargmax(autocepstrum_vals )
                    autocep_loss_val = torch.clamp(torch.abs(max_autocepstrum_val - symbols[i]), min = 0, max = 1)
                    curr_audio_num_err_symbols = curr_audio_num_err_symbols + autocep_loss_val
                    pred_symbols.append(int(max_autocepstrum_val.item()))
                    
                    # autocep_vals_nondiff = autocepstrum_vals.clone()
                    # autocep_vals_nondiff = autocep_vals_nondiff.detach().cpu().numpy()
                    # max_autocepstrum_val_nondiff = np.argmax(autocep_vals_nondiff)
                    # if max_autocepstrum_val_nondiff != symbols[i]:
                    #     tot_symbol_errors += 1 
                    
            # print(pred_symbols) 
            # print(symbols.detach().cpu().numpy().tolist())
            # print('-----------------')
            tot_symbol_errs = tot_symbol_errs + curr_audio_num_err_symbols
            if num_errs_reverb - num_errs_no_reverb == 0:
                curr_audio_err_reduction = 1
            else:
                curr_audio_err_reduction = ((num_errs_reverb - num_errs_no_reverb) - (curr_audio_num_err_symbols - num_errs_no_reverb)) / (num_errs_reverb - num_errs_no_reverb)
            total_err_reduction = total_err_reduction + (1 / curr_audio_err_reduction)

            gt_symbol_err_rate_no_reverb = gt_symbol_err_rate_no_reverb + num_errs_no_reverb
            gt_symbol_err_rate_reverb = gt_symbol_err_rate_reverb + num_errs_reverb

        symbol_err_rate = tot_symbol_errs / (audio_batch.shape[0] * num_wins)
        avg_err_reduction = total_err_reduction / audio_batch.shape[0]
        gt_symbol_err_rate_no_reverb = gt_symbol_err_rate_no_reverb / (audio_batch.shape[0] * num_wins)
        gt_symbol_err_rate_reverb = gt_symbol_err_rate_reverb / (audio_batch.shape[0] * num_wins)
        
        return symbol_err_rate, tot_symbol_errs, avg_err_reduction, gt_symbol_err_rate_no_reverb, gt_symbol_err_rate_reverb

    def softargmax(self, x):
        """
        beta original 1e10
        From StackOverflow user Lostefra
        https://stackoverflow.com/questions/46926809/getting-around-tf-argmax-which-is-not-differentiable
        """
        x_range = torch.arange(x.shape[-1], dtype=x.dtype, device = x.device)
        return torch.sum(torch.nn.functional.softmax(x*self.softargmax_beta, dim=-1) * x_range, dim=-1)


    def autocorrelation_1d(self, signal):
        """
        Computes the autocorrelation of a 1D signal using PyTorch.

        Args:
            signal (torch.Tensor): A 1D tensor representing the input signal.

        Returns:
            torch.Tensor: A 1D tensor representing the autocorrelation of the signal.
        """
        signal_length = signal.size(0)
        padded_signal = torch.nn.functional.pad(signal, (0, signal_length), mode='constant', value=0)
        
        # Reshape for convolution
        signal_reshaped = signal.reshape(1, 1, -1)
        padded_signal_reshaped = padded_signal.reshape(1, 1, -1)

        # Perform convolution (which is equivalent to cross-correlation for autocorrelation)
        autocorr = torch.nn.functional.conv1d(padded_signal_reshaped, signal_reshaped, padding=signal_length - 1)

        return autocorr.squeeze()


    def torch_get_cepstrum(self, signal):
        """
        Get the cepstrum of a signal in differentiable fashion using torch.

        Parameters:
            signal : torch.Tensor
        """
        fft = torch.fft.rfft(signal)
        sqr_log_fft = torch.log(fft.abs() + 0.00001)
        cepstrum = torch.fft.irfft(sqr_log_fft)

        # sanity check to make sure torch implementation is correct
        # test_fft = np.fft.fft(signal.numpy())
        # test_sqr_log_fft = np.log(np.abs(test_fft) + 0.00001)
        # test_cepstrum = np.fft.ifft(test_sqr_log_fft).real
        # print(cepstrum)
        # print(test_cepstrum)
        # print(np.allclose(cepstrum.numpy(), test_cepstrum, atol=1e-3))
        # print("---------------")
        return cepstrum

    def torch_get_autocepstrum(self, signal):
        """
        Get the autocepstrum of a signal in differentiable fashion using torch.
        """
        autocorr = self.autocorrelation_1d(signal)
        cep_autocorr = torch.fft.ifft(torch.log(torch.abs(torch.fft.fft(autocorr)) + 0.00001)).real
        return cep_autocorr
