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
    

    def forward(self, audio_batch, gt_symbols_batch, num_errs_no_reverb_batch, num_errs_reverb_batch):
        """
        Compute various loss functions for the decoding task.

        TODO: Ceptral loss

        Parameters:
            audio_batch : torch.Tensor (batch_size, 1, num_samples)
                Batch of audio samples in time domain
            gt_symbols_batch : torch.Tensor (batch_size, num_symbols) 
                Batch of groudn-truth symbols that were encoded onto the clean speech
            num_errs_no_reverb_batch : torch.Tensor (batch_size)
                Batch of number of errors when decoding the encoded clean speech (these can occur due to confounding peaks in speech 
                cepstra and are independent of the reverb or network)
            num_errs_reverb_batch : torch.Tensor (batch_size)
                Batch of number of errors when decoding the encoded reverb speech 
        
        """
     
        num_wins = audio_batch.shape[2] // self.win_size
        max_audio_len = num_wins * self.win_size

        # chop up audio into audio_batch.shape[2] // win_size windows organized along batch dimension
        audio_batch = audio_batch.squeeze(1) # (batch_size, 1, num_samples) -> (batch_size, num_samples)
        audio_batch = audio_batch[:, :max_audio_len] # preemptively cut so all splis will be equal
        res = torch.tensor_split(audio_batch, num_wins, dim=1) # (batch_size, num_samples) -> tuple of  num_wins tensors of shape (batch_size, win_size)
        audio_batch = torch.stack(res, dim=1) # tuple-> (batch_size, num_wins, win_size)
        all_windows = audio_batch.reshape(audio_batch.shape[0] * audio_batch.shape[1], self.win_size) # (batch_size, num_wins, win_size) -> (batch_size * num_wins, win_size)
        
        # compute symbols for all windows 
        all_pred_symbols = torch.vmap(self.compute_symbol)(all_windows) # apply compute_symbol to each window in all_windows
  
        # compute avg error rate
        all_gt_symbols = gt_symbols_batch.reshape(-1) # (batch_size, num_symbols) -> (batch_size * num_wins)
        accuracy = Accuracy(task="multiclass", num_classes=len(self.delays)).to(audio_batch.device)
        sym_err_rate = 1 - accuracy(all_pred_symbols, all_gt_symbols) 

        # compute err reduction loss, defined as the current number of errors per audio sample in the batch 
        # divided by the number of errors per input reverberant audio sample, normed by the num errs in the non-reverb audio, 
        # as those errors are "inevitable" and unrelated to the reverb 
        all_pred_symbols = all_pred_symbols.reshape(audio_batch.shape[0], num_wins) # (batch_size * num_wins, 1) -> (batch_size, num_wins)
        samplewise_accuracy = Accuracy(task="multiclass", num_classes=len(self.delays), multidim_average='samplewise').to(audio_batch.device) 
        num_err_per_samp = (1 - samplewise_accuracy(all_pred_symbols, gt_symbols_batch)) * num_wins
        norm_num_err_per_samp_premodel = torch.clamp(num_errs_reverb_batch - num_errs_no_reverb_batch, min = 1)
        err_reduction_loss_per_samp = torch.clamp(num_err_per_samp - num_errs_no_reverb_batch, min = 0) / norm_num_err_per_samp_premodel
        avg_err_reduction_loss = torch.mean(err_reduction_loss_per_samp)

        # compute ground truth symbol error rate for reverb and non-reverb, for logging purposes
        no_reverb_sym_err_rate = torch.sum(num_errs_no_reverb_batch) / (audio_batch.shape[0] * num_wins)
        reverb_sym_err_rate = torch.sum(num_errs_reverb_batch) / (audio_batch.shape[0] * num_wins)

        # print("Fast", sym_err_rate, avg_err_reduction_loss, no_reverb_sym_err_rate, reverb_sym_err_rate)
        return sym_err_rate, avg_err_reduction_loss, no_reverb_sym_err_rate, reverb_sym_err_rate

     
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
