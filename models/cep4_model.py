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

from decoding import CepstralDomainDecodingLoss



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
    
class TestAttention(pl.LightningModule):
    def __init__(self,
        learning_rate=1e-3,
        nwins=16,
        use_transformer=True, 
        alphas = [0, 0, 0, 0, 0],
        softargmax_beta = 100000,
        residual = False,
        delays = None,
        win_size = None,
        cutoff_freq = None,
        sample_rate = None,
        plot_every_n_steps=100,
        norm_cepstra = True,
        cepstrum_target_region=None
        ):
        super().__init__()
        self.name = "Test"

        self.has_init = False
        self.learning_rate = learning_rate
        self.plot_every_n_steps = plot_every_n_steps
        self.nwins = nwins
        self.use_transformer = use_transformer
        self.residual = residual
        self.norm_cepstra = norm_cepstra
        self.alphas = alphas
        if cepstrum_target_region is not None:
            self.cepstrum_target_region = cepstrum_target_region
        else:
            self.cepstrum_target_region = [delays[0] - 10, delays[-1] + 50] # default

        self.lr_scheduler_gamma = 0.9
        self.eps = 1e-16

        #initialize encoding parameters and decoding loss
        self.delays = delays
        self.win_size = win_size
        self.cutoff_freq = cutoff_freq
        self.sample_rate = sample_rate
        self.decoding_loss = CepstralDomainDecodingLoss(delays, 
                                          win_size, 
                                          cutoff_freq, 
                                          sample_rate, 
                                          softargmax_beta)

        self.init()

    
    def init(self, in_channels=2, out_channels=2, base_c=64):
        # Encoder
        self.enc1 = DoubleConv(in_channels, base_c, dilation=1)
        self.enc2 = DoubleConv(base_c, base_c*2, dilation=2)
        self.enc3 = DoubleConv(base_c*2, base_c*4, dilation=4)
        self.enc4 = DoubleConv(base_c*4, base_c*8, dilation=8)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_c*8, base_c*16)

        # Attention blocks
        self.att4 = AttentionBlock(F_g=base_c*8, F_l=base_c*8, F_int=base_c*4)
        self.att3 = AttentionBlock(F_g=base_c*4, F_l=base_c*4, F_int=base_c*2)
        self.att2 = AttentionBlock(F_g=base_c*2, F_l=base_c*2, F_int=base_c)
        self.att1 = AttentionBlock(F_g=base_c, F_l=base_c, F_int=base_c//2)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_c*16, base_c*8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_c*16, base_c*8)

        self.up3 = nn.ConvTranspose2d(base_c*8, base_c*4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_c*8, base_c*4)

        self.up2 = nn.ConvTranspose2d(base_c*4, base_c*2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_c*4, base_c*2)

        self.up1 = nn.ConvTranspose2d(base_c*2, base_c, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_c*2, base_c)

        self.final_conv = nn.Conv2d(base_c, out_channels, kernel_size=1)


    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with attention
        d4 = self.up4(b)
        e4 = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        e3 = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        e2 = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        e1 = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final_conv(d1), b

    def training_step(self, batch, batch_idx):
        loss_type = "train"
        enc_speech_cepstra, enc_reverb_speech_cepstra, unenc_reverb_speech_cepstra, \
                enc_speech_wav, enc_reverb_speech_wav, unenc_reverb_speech_wav, \
                rir, stochastic_noise, noise_condition, symbols,  idx_rir, num_errs_no_reverb, num_errs_reverb = batch
        enc_reverb_speech_cepstra = enc_reverb_speech_cepstra.float()
        enc_speech_cepstra = enc_speech_cepstra.float()
        unenc_reverb_speech_cepstra = unenc_reverb_speech_cepstra.float()
       
        # enc_reverb_speech_cepstra = enc_reverb_speech_cepstra[:,:,:512,:]   
        #  
        enc_speech_cepstra_hat, enc_b = self(enc_reverb_speech_cepstra) 
        _, unenc_b = self(unenc_reverb_speech_cepstra)
  
        # append zeros at the second dimension to get back to same size as enc_speech_cepstra (in case it was previously cropped)
        enc_speech_cepstra_hat = t.cat((enc_speech_cepstra_hat, t.zeros(enc_speech_cepstra.shape[0], enc_speech_cepstra.shape[1], enc_speech_cepstra.shape[2]-enc_speech_cepstra_hat.shape[2],enc_speech_cepstra.shape[3]).to(enc_speech_cepstra_hat)), dim=2)
        enc_reverb_speech_cepstra = t.cat((enc_reverb_speech_cepstra, t.zeros(enc_speech_cepstra.shape[0], enc_speech_cepstra.shape[1], enc_speech_cepstra.shape[2]-enc_reverb_speech_cepstra.shape[2],enc_speech_cepstra.shape[3] ).to(enc_reverb_speech_cepstra)), dim=2)
        
        if batch_idx % self.plot_every_n_steps == 0:
            plot = True
        else:
            plot = False

        bottleneck_mse = nn.functional.mse_loss(enc_b, unenc_b)
        cepstra_loss, full_cepstra_loss = self.compute_speech_loss(enc_speech_cepstra, enc_speech_cepstra_hat ) 
        sym_err_rate, avg_err_reduction_loss, no_reverb_sym_err_rate, reverb_sym_err_rate  = self.decoding_loss(enc_speech_cepstra_hat, symbols, num_errs_no_reverb, num_errs_reverb)
        loss = self.alphas[0] * cepstra_loss + self.alphas[1] * sym_err_rate + full_cepstra_loss * self.alphas[2]  + self.alphas[3] * avg_err_reduction_loss + bottleneck_mse * 2
        
        self.log(loss_type+"_cep_mse_loss", cepstra_loss, sync_dist = True )
        self.log(loss_type+"_sym_err_rate", sym_err_rate, sync_dist = True )
        self.log(loss_type+"_full_cep_mse_loss", full_cepstra_loss, sync_dist = True )
        self.log(loss_type+"_bottleneck_mse", bottleneck_mse, sync_dist = True )
        self.log(loss_type+"_avg_err_reduction_loss", avg_err_reduction_loss, sync_dist = True )
        self.log(loss_type+"_no_reverb_sym_err_rate", no_reverb_sym_err_rate, sync_dist = True )
        self.log(loss_type+"_reverb_sym_err_rate", reverb_sym_err_rate, sync_dist = True )
        self.log(loss_type+"_loss", loss, sync_dist = True )
      
        if plot:
            self.make_cepstra_plot(enc_reverb_speech_cepstra, enc_speech_cepstra, enc_speech_cepstra_hat, symbols)
            # self.weight_histograms()
            
        return loss

    def validation_step(self, batch, batch_idx):
        loss_type = "val"

        enc_speech_cepstra, enc_reverb_speech_cepstra, unenc_reverb_speech_cepstra, \
                enc_speech_wav, enc_reverb_speech_wav, unenc_reverb_speech_wav, \
                rir, stochastic_noise, noise_condition, symbols,  idx_rir, num_errs_no_reverb, num_errs_reverb = batch
        enc_reverb_speech_cepstra = enc_reverb_speech_cepstra.float()
        enc_speech_cepstra = enc_speech_cepstra.float()

        # enc_reverb_speech_cepstra = enc_reverb_speech_cepstra[:,:,:512,:]
   
        enc_speech_cepstra_hat, _ = self(enc_reverb_speech_cepstra) 

        # append zeros at the second dimension to get back to same size as enc_speech_cepstra
        enc_speech_cepstra_hat = t.cat((enc_speech_cepstra_hat, t.zeros(enc_speech_cepstra.shape[0], enc_speech_cepstra.shape[1], enc_speech_cepstra.shape[2]-enc_speech_cepstra_hat.shape[2],enc_speech_cepstra.shape[3] ).to(enc_speech_cepstra_hat)), dim=2)
        enc_reverb_speech_cepstra = t.cat((enc_reverb_speech_cepstra, t.zeros(enc_speech_cepstra.shape[0], enc_speech_cepstra.shape[1], enc_speech_cepstra.shape[2]-enc_reverb_speech_cepstra.shape[2],enc_speech_cepstra.shape[3] ).to(enc_reverb_speech_cepstra)), dim=2)

        print(enc_speech_cepstra_hat.shape, symbols.shape, num_errs_no_reverb.shape, num_errs_reverb.shape)
        
        cepstra_loss, full_cepstra_loss = self.compute_speech_loss(enc_speech_cepstra, enc_speech_cepstra_hat ) 
        sym_err_rate, avg_err_reduction_loss, no_reverb_sym_err_rate, reverb_sym_err_rate  = self.decoding_loss(enc_speech_cepstra_hat, symbols, num_errs_no_reverb, num_errs_reverb)
        loss = self.alphas[0] * cepstra_loss + self.alphas[1] * sym_err_rate + full_cepstra_loss * self.alphas[2]  + self.alphas[3] * avg_err_reduction_loss 
        
        self.log(loss_type+"_cep_mse_loss", cepstra_loss, sync_dist = True )
        self.log(loss_type+"_sym_err_rate", sym_err_rate, sync_dist = True )
        self.log(loss_type+"_full_cep_mse_loss", full_cepstra_loss, sync_dist = True )
        self.log(loss_type+"_avg_err_reduction_loss", avg_err_reduction_loss, sync_dist = True )
        self.log(loss_type+"_no_reverb_sym_err_rate", no_reverb_sym_err_rate, sync_dist = True )
        self.log(loss_type+"_reverb_sym_err_rate", reverb_sym_err_rate, sync_dist = True )
        self.log(loss_type+"_loss", loss, sync_dist = True )
      
        self.make_cepstra_plot(enc_reverb_speech_cepstra, enc_speech_cepstra, enc_speech_cepstra_hat, symbols)

        return loss

    def test_step(self, batch, batch_idx):
        loss_type = "test"
        # TODO
        # return loss

    def compute_speech_loss(self, y, y_predict):
        """
        Compute all loss functions potentially incorporated during training.
        Inputs can be of speech or RIR.

        Args:
            y: 
                ground-truth frequency-domain representation (RIR or speech)
            yt: 
                ground-truth time-domain representation (RIR or speech)
            y_predict:  
                predicted frequency-domain representation (RIR or speech)
            type: str 
                "train", "val", or "test"
            
        Returns:
            All of the computed losses
        """
        # only examine encoding-relevant region of cepstra, as a large portion of it beyond this region is close to zero
        # considering this largely zero-valued tail causes the MSE to be close to zero all the time
        p_hat = y_predict[:,0, self.cepstrum_target_region[0]:self.cepstrum_target_region[1],:] 
        p = y[:,0, self.cepstrum_target_region[0]:self.cepstrum_target_region[1],:]
        if self.norm_cepstra:
            # minmax scasle target region
            p_mins = p.min(dim = 1, keepdim=True)[0]        
            p_maxs = p.max(dim = 1, keepdim=True)[0]
            p = (p - p_mins) / (p_maxs - p_mins)
            p_hat_mins = p_hat.min(dim = 1, keepdim=True)[0]
            p_hat_maxs = p_hat.max(dim = 1, keepdim=True)[0]
            p_hat = (p_hat - p_hat_mins) / (p_hat_maxs - p_hat_mins)
            p_hat[torch.isinf(p_hat)] = 0
            p[torch.isinf(p)] = 0
            p_hat[torch.isnan(p_hat)] = 0
            p[torch.isnan(p)] = 0
        mse_abs = nn.functional.mse_loss(p_hat, p)

        p_hat_full = y_predict[:,0,:,:]
        p_full = y[:,0,:,:]
        if self.norm_cepstra:
            # minmax scasle target region
            p_mins = p_full.min(dim = 1, keepdim=True)[0]        
            p_maxs = p_full.max(dim = 1, keepdim=True)[0]
            p_full = (p_full - p_mins) / (p_maxs - p_mins)
            p_hat_mins = p_hat_full.min(dim = 1, keepdim=True)[0]
            p_hat_maxs = p_hat_full.max(dim = 1, keepdim=True)[0]
            p_hat_full = (p_hat_full - p_hat_mins) / (p_hat_maxs - p_hat_mins)
            p_hat_full[torch.isinf(p_hat_full)] = 0
            p_full[torch.isinf(p_full)] = 0
            p_hat_full[torch.isnan(p_hat_full)] = 0
            p_full[torch.isnan(p_full)] = 0
        mse_abs_full = nn.functional.mse_loss(p_hat_full, p_full)
        return mse_abs, mse_abs_full
    

    def make_cepstra_plot(self, input_cepstra, clean_cepstra, pred_cepstra, symbols, loss_type = "train"):
        fh = plt.figure()
        fig, axes = plt.subplots(3, 4, figsize=(12, 5), tight_layout=True)
        batch_el = np.random.randint(0, input_cepstra.shape[0])
        window_start = np.random.randint(0, len(symbols[0])-4)
        for i in range(window_start, window_start+4):
            xvals = np.arange(self.cepstrum_target_region[0], self.cepstrum_target_region[1])
            axes[0, i - window_start].plot(xvals, input_cepstra[batch_el, 0,  self.cepstrum_target_region[0]:self.cepstrum_target_region[1] , i].detach().cpu().numpy())
            axes[0, i - window_start].set_title(f"Input cepstrum - Samp {batch_el} Win{i}")
            axes[1, i - window_start].plot(xvals, clean_cepstra[batch_el, 0,  self.cepstrum_target_region[0]:self.cepstrum_target_region[1] , i].detach().cpu().numpy())
            axes[1, i - window_start].set_title(f"Clean cepstrum - Samp {batch_el} Win{i}")
            axes[2, i - window_start].plot(xvals, pred_cepstra[batch_el, 0,  self.cepstrum_target_region[0]:self.cepstrum_target_region[1], i].detach().cpu().numpy()) 
            axes[2, i - window_start].set_title(f"Pred. cepstrum - Samp {batch_el} Win{i}")
        
        tb = self.logger.experiment
        tb.add_figure(loss_type + 'Cepstra', fig, global_step=self.global_step)
        plt.close()

        # make another version of the figure but without zooming in
        fh = plt.figure()
        fig, axes = plt.subplots(3, 4, figsize=(12, 5), tight_layout=True)
        for i in range(window_start, window_start+4):
            axes[0, i - window_start].plot(input_cepstra[batch_el, 0, : , i].detach().cpu().numpy())
            axes[0, i - window_start].set_title(f"Input cepstrum - Samp {batch_el} Win{i}")
            axes[1, i - window_start].plot(clean_cepstra[batch_el, 0,  : , i].detach().cpu().numpy())
            axes[1, i - window_start].set_title(f"Clean cepstrum - Samp {batch_el} Win{i}")
            axes[2, i - window_start].plot(pred_cepstra[batch_el, 0,  :, i].detach().cpu().numpy()) 
            axes[2, i - window_start].set_title(f"Pred. cepstrum - Samp {batch_el} Win{i}")
        
        tb = self.logger.experiment
        tb.add_figure(loss_type + 'FullCepstra', fig, global_step=self.global_step)
        plt.close()
  
        

    def make_rir_plot(self, gt_rir, pred_rir, loss_type = "train", intrabatch_rir_mse = None):
        fh = plt.figure()
        fig, axes = plt.subplots(2, 4, figsize=(12, 5), tight_layout=True)
        # randomly choose for samples from the batch
        batch_els = np.random.randint(0, gt_rir.shape[0], 4)
        for i, batch_el in enumerate(batch_els):
            axes[0, i].plot(gt_rir[batch_el, :].detach().cpu().numpy(), alpha = 0.5, label = "gt")
            axes[0, i].plot(pred_rir[batch_el, :].detach().cpu().numpy(), alpha = 0.5, label = "pred")
            axes[0, i].set_title(f"RIR Samp {batch_el}")
            axes[0, i].legend()

            axes[1, i].plot(gt_rir[batch_el, :].detach().cpu().numpy(), alpha = 0.5, label = "gt")
            axes[1, i].plot(pred_rir[batch_el, :].detach().cpu().numpy(), alpha = 0.5, label = "pred")
            axes[1, i].set_xlim(0, 1000)
            axes[1, i].set_title(f"(Zoomed in) RIR Samp {batch_el}")
            axes[1, i].legend()

            if intrabatch_rir_mse is not None:
                plt.suptitle(f"Intrabatch RIR MSE: {intrabatch_rir_mse:.4f}")

        tb = self.logger.experiment
        tb.add_figure(loss_type + 'RIR', fig, global_step=self.global_step)
        plt.close()

    def weight_histograms_conv2d(self, writer, step, weights, layer_number):
        weights_shape = weights.shape
        num_kernels = weights_shape[0]
        for k in range(num_kernels):
            flattened_weights = weights[k].flatten()
            tag = f"layer_{layer_number}/kernel_{k}"
            tb = self.logger.experiment
            tb.add_histogram(tag, flattened_weights, global_step=step, bins='tensorflow')

    def weight_histograms(self):
        if torch.utils.data.get_worker_info() is None:
            writer = self.logger.experiment
            step = self.global_step
            # Iterate over all model layers
            layers = []
            layers.append(self.conv1[0])
            layers.append(self.conv2[0])
            layers.append(self.conv3[0])
            layers.append(self.conv4[0])
            # layers.append(self.conv5[0])
            if self.use_transformer:
                layers.append(self.transformer)
            layers.append(self.deconv1[0])
            layers.append(self.deconv2[0])
            layers.append(self.deconv3[0])
            layers.append(self.deconv4[0])
            # layers.append(self.deconv5[0])
            
            for layer_number in range(len(layers)):
                layer = layers[layer_number]
                # Compute weight histograms for appropriate layer
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                    weights = layer.weight
                    self.weight_histograms_conv2d(writer, step, weights, layer_number)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        # return optimizer
        scheduler = lr_scheduler.ExponentialLR(optimizer, self.lr_scheduler_gamma, self.current_epoch-1)
        return [optimizer], [scheduler]

    