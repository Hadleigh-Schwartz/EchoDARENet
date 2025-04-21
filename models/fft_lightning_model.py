from torch import optim, nn
from torch.optim import lr_scheduler
import pytorch_lightning as pl
import torch as t
from torch.autograd import Function
import torch.utils.data
import torchaudio as ta

import matplotlib.pyplot as plt
import numpy as np
import math
from colorama import Fore, Style

from decoding import TimeDomainDecodingLoss

def getModel(model_name=None, learning_rate = 1e-3,  use_transformer = False, alphas = [0, 0, 0, 0, 0], softargmax_beta = 100000,
            delays = None, win_size = None, cutoff_freq = None, sample_rate = None, reverse_gradient = False, plot_every_n_steps=100,
            fft_target_region=None):
    if model_name == "EchoSpeechDAREUnet": model = EchoSpeechDAREUnet(learning_rate = learning_rate, 
                                                                    use_transformer = use_transformer, alphas = alphas, softargmax_beta = softargmax_beta,
                                                                    delays = delays, win_size = win_size, cutoff_freq = cutoff_freq, sample_rate = sample_rate, 
                                                                    reverse_gradient = reverse_gradient, plot_every_n_steps=plot_every_n_steps, fft_target_region = fft_target_region)    
    else: raise Exception("Unknown model name.")
    return model

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class EchoSpeechDAREUnet(pl.LightningModule):
    def __init__(self,
        learning_rate=1e-3,
        use_transformer=True, 
        alphas = [0, 0, 0, 0, 0],
        softargmax_beta = 100000,
        delays = None,
        win_size = None,
        cutoff_freq = None,
        sample_rate = None,
        reverse_gradient = False,
        plot_every_n_steps=100,
        fft_target_region=[0, 1000]
        ):
        super().__init__()
        self.name = "EchoSpeechDAREUnet"

        self.has_init = False
        self.learning_rate = learning_rate
        self.reverse_gradient = reverse_gradient
        self.plot_every_n_steps = plot_every_n_steps
        self.use_transformer = use_transformer
        self.alphas = alphas
        self.fft_target_region = fft_target_region
      

        self.lr_scheduler_gamma = 0.9
        self.eps = 1e-16

        #initialize encoding parameters and decoding loss
        self.delays = delays
        self.win_size = win_size
        self.cutoff_freq = cutoff_freq
        self.sample_rate = sample_rate
        self.decoding_loss = TimeDomainDecodingLoss(delays,
                                                    win_size,
                                                    "cepstrum",
                                                    cutoff_freq,
                                                    sample_rate,
                                                    softargmax_beta)
        self.init()

    def init(self):
        k = 5
        s = 2
        p_drop = 0.5
        leaky_slope = 0.01
        
        self.conv1 = nn.Sequential(nn.Conv1d(  2,  64, k, stride=s, padding=k//2), nn.LeakyReLU(leaky_slope))
        self.conv2 = nn.Sequential(nn.Conv1d( 64, 128, k, stride=s, padding=k//2), nn.BatchNorm1d(128), nn.LeakyReLU(leaky_slope))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 256, k, stride=s, padding=k//2), nn.BatchNorm1d(256), nn.LeakyReLU(leaky_slope))
        self.conv4 = nn.Sequential(nn.Conv1d(256, 256, k, stride=s, padding=k//2), nn.BatchNorm1d(256), nn.ReLU())
        
        if self.use_transformer:
            self.transformer = nn.Transformer(d_model=256, nhead=4, num_encoder_layers=5, num_decoder_layers=5, batch_first=True)

        self.deconv1 = nn.Sequential(nn.ConvTranspose1d(256, 256, k, stride=s, padding=k//2), nn.BatchNorm1d(256), nn.Dropout1d(p=p_drop), nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose1d(512, 128, k, stride=s, padding=k//2, output_padding=s-1), nn.BatchNorm1d(128), nn.Dropout1d(p=p_drop), nn.ReLU())
        self.deconv3 = nn.Sequential(nn.ConvTranspose1d(256,  64, k, stride=s, padding=k//2, output_padding=s-1), nn.BatchNorm1d(64),  nn.ReLU())
        self.deconv4 = nn.Sequential(nn.ConvTranspose1d(128,   2, k, stride=s, padding=k//2, output_padding=s-1), nn.BatchNorm1d(2)) 

    def predict(self, x):

        c1Out = self.conv1(x)     # (64 x 64 x  64)
        c2Out = self.conv2(c1Out) # (16 x 16 x 128)
        c3Out = self.conv3(c2Out) # ( 4 x  4 x 256)
        c4Out = self.conv4(c3Out) # ( 1 x  1 x 256)

        if self.use_transformer:
            c4Out = self.transformer(\
                c4Out.squeeze().permute((0,2,1)), \
                c4Out.squeeze().permute((0,2,1))).permute((0,2,1))
        
        d1Out = self.deconv1(c4Out) # (  4 x   4 x 256)
        d2Out = self.deconv2(t.cat((d1Out, c3Out), dim=1)) # ( 16 x  16 x 128)
        d3Out = self.deconv3(t.cat((d2Out, c2Out), dim=1)) # ( 64 x  64 x 128)
        fftOut = self.deconv4(t.cat((d3Out, c1Out), dim=1)) # (256 x 256 x 1)
        return fftOut

    def training_step(self, batch, batch_idx):
        loss_type = "train"

        _, _, x_fft, y_fft, z_fft, reverb_speech_wav, speech_wav, symbols, num_errs_no_reverb, num_errs_reverb, idxs_rir, idxs_speech = batch # reverberant speech STFT, clean speech STFT, RIR fft, RIR time domain, symbols echo-encoded into speech, error rate of echo-decoding pre-reverb
        
        x_fft = x_fft.float()
        y_fft = y_fft.float()
        z_fft = z_fft.float()

        rev_fft_hat = self.predict(x_fft)
        rev2_fft_hat = self.predict(z_fft) 

        rev_fft_hat_c = rev_fft_hat[:,0,:] + 1j*rev_fft_hat[:,1,:]
        x_fft_c = x_fft[:,0,:] + 1j*x_fft[:,1,:]
        y_fft_c = y_fft[:,0,:] + 1j*y_fft[:,1,:]
        y_fft_hat_c = rev_fft_hat_c * x_fft_c
        y_fft_hat = t.stack((y_fft_hat_c.real, y_fft_hat_c.imag), dim=1)

        # compute inverse rfft of y_fft_hat_c
        y_t_hat = torch.fft.irfft(y_fft_hat_c, n =  30735, dim = 1)
        y_t_hat = y_t_hat.unsqueeze(1) # the decoding loss forward expects input of shape (batch_size, 1, num_samples)
        sym_err_rate, avg_err_reduction_loss, no_reverb_sym_err_rate, reverb_sym_err_rate  = self.decoding_loss(y_t_hat, symbols, num_errs_no_reverb, num_errs_reverb)
        
        # time domain speech loss
        assert y_t_hat.shape == speech_wav.shape
        time_loss = nn.functional.mse_loss(y_t_hat, speech_wav) * 500000
        complex_loss, two_channel_loss = self.compute_fft_loss(y_fft_c, y_fft_hat_c) 
        pair_loss = nn.functional.mse_loss(rev_fft_hat, rev2_fft_hat)
        loss = two_channel_loss + pair_loss + time_loss

        if batch_idx % self.plot_every_n_steps == 0:
            plot = True
        else:
            plot = False
        self.log(loss_type+"_time_loss", time_loss, sync_dist = True )
        self.log(loss_type+"_complex_loss", complex_loss, sync_dist = True )
        self.log(loss_type+"_two_channel_loss", two_channel_loss, sync_dist = True )
        self.log(loss_type+"_pair_loss", pair_loss, sync_dist = True )
        self.log(loss_type+"_sym_err_rate", sym_err_rate, sync_dist = True )
        self.log(loss_type+"_avg_err_reduction_loss", avg_err_reduction_loss, sync_dist = True )
        self.log(loss_type+"_no_reverb_sym_err_rate", no_reverb_sym_err_rate, sync_dist = True )
        self.log(loss_type+"_reverb_sym_err_rate", reverb_sym_err_rate, sync_dist = True )
        self.log(loss_type+"_loss", loss, sync_dist = True )

        if plot:
            self.make_fft_plot(x_fft, y_fft, y_fft_hat, loss_type)
            self.make_time_domain_plot(reverb_speech_wav, speech_wav, y_t_hat, loss_type)
            
        return loss

    def validation_step(self, batch, batch_idx):
        loss_type = "val"

        _, _, x_fft, y_fft,  z_fft, reverb_speech_wav, speech_wav, symbols, num_errs_no_reverb, num_errs_reverb, idxs_rir, idxs_speech = batch # reverberant speech STFT, clean speech STFT, RIR fft, RIR time domain, symbols echo-encoded into speech, error rate of echo-decoding pre-reverb
        
        x_fft = x_fft.float()
        y_fft = y_fft.float()
        z_fft = z_fft.float()

        rev_fft_hat = self.predict(x_fft)
        rev2_fft_hat = self.predict(z_fft)

        rev_fft_hat_c = rev_fft_hat[:,0,:] + 1j*rev_fft_hat[:,1,:]
        x_fft_c = x_fft[:,0,:] + 1j*x_fft[:,1,:]
        y_fft_c = y_fft[:,0,:] + 1j*y_fft[:,1,:]
        y_fft_hat_c = rev_fft_hat_c * x_fft_c
        y_fft_hat = t.stack((y_fft_hat_c.real, y_fft_hat_c.imag), dim=1)

        # compute inverse rfft of y_fft_hat_c
        y_t_hat = torch.fft.irfft(y_fft_hat_c, n =  30735, dim = 1)
        y_t_hat = y_t_hat.unsqueeze(1) # the decoding loss forward expects input of shape (batch_size, 1, num_samples)
        sym_err_rate, avg_err_reduction_loss, no_reverb_sym_err_rate, reverb_sym_err_rate  = self.decoding_loss(y_t_hat, symbols, num_errs_no_reverb, num_errs_reverb)
        
        # time domain speech loss
        time_loss = nn.functional.mse_loss(y_t_hat, speech_wav.unsqueeze(1)) * 500000
        complex_loss, two_channel_loss = self.compute_fft_loss(y_fft_c, y_fft_hat_c) 
        pair_loss = nn.functional.mse_loss(rev_fft_hat, rev2_fft_hat)
        loss = two_channel_loss + pair_loss + time_loss

        self.log(loss_type+"_time_loss", time_loss, sync_dist = True )
        self.log(loss_type+"_complex_loss", complex_loss, sync_dist = True )
        self.log(loss_type+"_two_channel_loss", two_channel_loss, sync_dist = True )
        self.log(loss_type+"_pair_loss", pair_loss, sync_dist = True )
        self.log(loss_type+"_sym_err_rate", sym_err_rate, sync_dist = True )
        self.log(loss_type+"_avg_err_reduction_loss", avg_err_reduction_loss, sync_dist = True )
        self.log(loss_type+"_no_reverb_sym_err_rate", no_reverb_sym_err_rate, sync_dist = True )
        self.log(loss_type+"_reverb_sym_err_rate", reverb_sym_err_rate, sync_dist = True )
        self.log(loss_type+"_loss", loss, sync_dist = True )


        self.make_fft_plot(x_fft, y_fft, y_fft_hat, loss_type)
        self.make_time_domain_plot(reverb_speech_wav, speech_wav, y_t_hat, loss_type)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        loss_type = "test"
        # TODO
        # return loss

    def compute_fft_loss(self, y_fft_c, y_fft_hat_c):
        """
        Compute all loss functions potentially incorporated during training.
        Inputs can be of speech or RIR.

        Args:
            y: 
                ground-truth FFT
            y_hat:  
                predicted  FFT
            type: str 
                "train", "val", or "test"
            
        Returns:
            All of the computed losses
        """
        # separate real+imag version
        err_real = nn.functional.l1_loss(t.real(y_fft_hat_c)[:,self.fft_target_region[0]:self.fft_target_region[1]],t.real(y_fft_c)[:,self.fft_target_region[0]:self.fft_target_region[1]])
        err_imag = nn.functional.l1_loss(t.imag(y_fft_hat_c)[:,self.fft_target_region[0]:self.fft_target_region[1]],t.imag(y_fft_c)[:,self.fft_target_region[0]:self.fft_target_region[1]])
        err_abs = nn.functional.l1_loss(t.log(t.abs(y_fft_hat_c))[:,self.fft_target_region[0]:self.fft_target_region[1]],t.log(t.abs(y_fft_c)[:,self.fft_target_region[0]:self.fft_target_region[1]]))
        y1 = t.sin(t.angle(y_fft_c))[:,self.fft_target_region[0]:self.fft_target_region[1]]
        y2 = t.cos(t.angle(y_fft_c))[:,self.fft_target_region[0]:self.fft_target_region[1]]
        y_hat1 = t.sin(t.angle(y_fft_hat_c))[:,self.fft_target_region[0]:self.fft_target_region[1]]
        y_hat2 = t.cos(t.angle(y_fft_hat_c))[:,self.fft_target_region[0]:self.fft_target_region[1]]
        err_phase = nn.functional.l1_loss(y1, y_hat1) + nn.functional.l1_loss(y2, y_hat2)
        complex_loss = err_real + err_imag
        
        # 2d channel version
        y_fft_hat = t.stack((y_fft_hat_c.real, y_fft_hat_c.imag), dim=1)
        y_fft = t.stack((y_fft_c.real, y_fft_c.imag), dim=1)

        # only examine encoding-relevant region of FFT, as a large portion of it beyond this region is close to zero
        # considering this largely zero-valued tail causes the MSE to be close to zero all the time
        p_hat = y_fft_hat[:,:, self.fft_target_region[0]:self.fft_target_region[1]] 
        p = y_fft[:,:, self.fft_target_region[0]:self.fft_target_region[1]]
        two_channel_loss = nn.functional.mse_loss(p_hat, p)

        return complex_loss, two_channel_loss

    def make_fft_plot(self, input_fft, clean_fft, pred_fft, loss_type = "train"):
        fh = plt.figure()
        fig, axes = plt.subplots(3, 4, figsize=(12, 5), tight_layout=True)
        plot_els = np.random.randint(0, input_fft.shape[0], 4)
        for i, batch_el in enumerate(plot_els):
            
            axes[0, i].plot(input_fft[batch_el, 0, self.fft_target_region[0]:self.fft_target_region[1]].detach().cpu().numpy(), label = "Real", alpha = 0.5)
            axes[0, i].plot(input_fft[batch_el, 1, self.fft_target_region[0]:self.fft_target_region[1]].detach().cpu().numpy(), label = "Imag",  alpha = 0.5)
            axes[0, i].set_title(f"Input FFT - Samp {batch_el}")
            axes[0, i].legend()
            axes[1, i].plot(clean_fft[batch_el, 0, self.fft_target_region[0]:self.fft_target_region[1]].detach().cpu().numpy(), label = "Real",  alpha = 0.5)
            axes[1, i].plot(clean_fft[batch_el, 1, self.fft_target_region[0]:self.fft_target_region[1]].detach().cpu().numpy(), label = "Imag",  alpha = 0.5)
            axes[1, i].set_title(f"Clean FFT - Samp {batch_el}")
            axes[1, i].legend()
            axes[2, i].plot(pred_fft[batch_el, 0, self.fft_target_region[0]:self.fft_target_region[1]].detach().cpu().numpy(), label = "Real",  alpha = 0.5)
            axes[2, i].plot(pred_fft[batch_el, 1, self.fft_target_region[0]:self.fft_target_region[1]].detach().cpu().numpy(), label = "Imag",  alpha = 0.5)
            axes[2, i].set_title(f"Pred. FFT - Samp {batch_el}")
            axes[2, i].legend()
        
        tb = self.logger.experiment
        tb.add_figure(loss_type + 'FFT', fig, global_step=self.global_step)
        plt.close()

    def make_time_domain_plot(self, input_wav, clean_wav, pred_wav, loss_type = "train"):
        fh = plt.figure()
        fig, axes = plt.subplots(3, 4, figsize=(12, 5), tight_layout=True)
        plot_els = np.random.randint(0, input_wav.shape[0], 4)
        for i, batch_el in enumerate(plot_els):
            axes[0, i].plot(input_wav[batch_el, 0, :].detach().cpu().numpy())
            axes[0, i].set_title(f"Input Wav - Samp {batch_el}")
            axes[1, i].plot(clean_wav[batch_el, 0, :].detach().cpu().numpy())
            axes[1, i].set_title(f"Clean Wav - Samp {batch_el}")
            axes[2, i].plot(pred_wav[batch_el, 0, :].detach().cpu().numpy())
            axes[2, i].set_title(f"Pred. Wav - Samp {batch_el}")

        tb = self.logger.experiment
        tb.add_figure(loss_type + 'wav', fig, global_step=self.global_step)
        plt.close()

    def make_cepstra_plot(self, input_cepstra, clean_cepstra, pred_cepstra, symbols, loss_type = "train"):
        # TODO: update
        fh = plt.figure()
        fig, axes = plt.subplots(3, 4, figsize=(12, 5), tight_layout=True)
        batch_el = np.random.randint(0, input_cepstra.shape[0])
        window_start = np.random.randint(0, len(symbols[0])-4)
        for i in range(window_start, window_start+4):
            axes[0, i - window_start].plot(input_cepstra[batch_el, 0, self.delays[0] - 10:self.delays[-1] + 50 , i].detach().cpu().numpy())
            axes[0, i - window_start].set_title(f"Input cepstrum - Samp {batch_el} Win{i}")
            axes[1, i - window_start].plot(clean_cepstra[batch_el, 0, self.delays[0] - 10:self.delays[-1] + 50 , i].detach().cpu().numpy())
            axes[1, i - window_start].set_title(f"Clean cepstrum - Samp {batch_el} Win{i}")
            axes[2, i - window_start].plot(pred_cepstra[batch_el, 0,  self.delays[0] - 10:self.delays[-1] + 50, i].detach().cpu().numpy()) 
            axes[2, i - window_start].set_title(f"Pred. cepstrum - Samp {batch_el} Win{i}")
        
        tb = self.logger.experiment
        tb.add_figure(loss_type + 'Cepstra', fig, global_step=self.global_step)
        plt.close()

  
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        # return optimizer
        scheduler = lr_scheduler.ExponentialLR(optimizer, self.lr_scheduler_gamma, self.current_epoch-1)
        return [optimizer], [scheduler]

    