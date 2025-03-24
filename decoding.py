import torch
#import torch.nn as nn
from torch import optim, nn
import pytorch_lightning as pl
from torchaudio.functional import highpass_biquad
from torchmetrics import Accuracy

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

    def compute_symbol(self, audio_window):
        """
        Compute the symbol for a given audio window.
        Parameters:
            audio_window : torch.Tensor (win_size)
                Window of audio samples in time domain
        Returns:
            max_val : float
                The decoded symbol for the given audio window
        """
        if self.decoding == "cepstrum":
            cepstrum = self.torch_get_cepstrum(audio_window)
            cep_vals = cepstrum[self.delays]
            cep_decoded = self.softargmax(cep_vals)
            return cep_decoded
        else:
            autocepstrum = self.torch_get_autocepstrum(audio_window)
            autocep_vals = autocepstrum[self.delays]
            autocep_decoded = self.softargmax(autocep_vals)
            return autocep_decoded
    

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
        
        # call torch.vmap on X to get a tensor Y of shape (batch_size * num_wins, 1) of symbols
        all_symbols = torch.vmap(self.compute_symbol)(all_windows) # apply compute_symbol to each window in all_windows

        # reshape Y into (batch_size, num_wins)
        # reshape symbols_batch into (batch_size * num_wins, 1)
        all_gt_symbols = symbols_batch.reshape(-1) # (batch_size, num_symbols) -> (batch_size * num_wins)
 
        # compute avg error rate of all symbols and all_gt_symbols
        accuracy = Accuracy(task="multiclass", num_classes=len(self.delays))
        sym_err_rate = 1 - accuracy(all_symbols, all_gt_symbols) 

        print("Fast", sym_err_rate)

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
        tot_sym_err = 0 # total symbol errors (differentiable)
        tot_err_reduction = 0 # accrue total error reduction (differentiable)
        gt_symbol_err_rate_reverb = 0
        gt_symbol_err_rate_no_reverb = 0

        num_wins = audio_batch.shape[2] // self.win_size
        max_audio_len = num_wins * self.win_size
        for audio_idx in range(audio_batch.shape[0]):
            audio = audio_batch[audio_idx, 0, :max_audio_len]
            symbols = symbols_batch[audio_idx]
            num_errs_no_reverb = num_errs_no_reverb_batch[audio_idx]
            num_errs_reverb = num_errs_reverb_batch[audio_idx]
           
            curr_audio_tot_sym_err = 0
            # pred_symbols = []

            if self.cutoff_freq is not None: # high-pass filter the window 
                audio = highpass_biquad(audio,  self.cutoff_freq , self.sample_rate) # FILTERING IS WRONG???
            for i in range(num_wins):
                win = audio[i * self.win_size: (i + 1) * self.win_size]

                if self.decoding == "cepstrum":
                    cepstrum = self.torch_get_cepstrum(win)
                    cep_vals = cepstrum[self.delays]
                    cep_decoded = self.softargmax(cep_vals)
                    cep_sym_err = torch.clamp(torch.abs(cep_decoded - symbols[i]), min = 0, max = 1)
                    curr_audio_tot_sym_err = curr_audio_tot_sym_err + cep_sym_err
                    # pred_symbols.append(int(cep_decoded.item()))

                else:
                    autocepstrum = self.torch_get_autocepstrum(win)
                    autocepstrum_vals  = autocepstrum[self.delays]
                    max_autocepstrum_val  = self.softargmax(autocepstrum_vals )
                    autocep_sym_err = torch.clamp(torch.abs(max_autocepstrum_val - symbols[i]), min = 0, max = 1)
                    curr_audio_tot_sym_err = curr_audio_tot_sym_err + autocep_sym_err
                    # pred_symbols.append(int(max_autocepstrum_val.item()))
                    
            # print(pred_symbols) 
            # print(symbols.detach().cpu().numpy().tolist())
            # print('-----------------')
            tot_sym_err = tot_sym_err + curr_audio_tot_sym_err
            if num_errs_reverb - num_errs_no_reverb == 0:
                curr_audio_err_reduction = 1
            else:
                curr_audio_err_reduction = ((num_errs_reverb - num_errs_no_reverb) - (curr_audio_tot_sym_err - num_errs_no_reverb)) / (num_errs_reverb - num_errs_no_reverb)
            tot_err_reduction = tot_err_reduction + (1 / curr_audio_err_reduction)

            gt_symbol_err_rate_no_reverb = gt_symbol_err_rate_no_reverb + num_errs_no_reverb
            gt_symbol_err_rate_reverb = gt_symbol_err_rate_reverb + num_errs_reverb

        sym_err_rate = tot_sym_err / (audio_batch.shape[0] * num_wins)
        avg_err_reduction = tot_err_reduction / audio_batch.shape[0]
        gt_symbol_err_rate_no_reverb = gt_symbol_err_rate_no_reverb / (audio_batch.shape[0] * num_wins)
        gt_symbol_err_rate_reverb = gt_symbol_err_rate_reverb / (audio_batch.shape[0] * num_wins)
        
        print("Forward", sym_err_rate)
        return sym_err_rate, tot_sym_err, avg_err_reduction, gt_symbol_err_rate_no_reverb, gt_symbol_err_rate_reverb

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
