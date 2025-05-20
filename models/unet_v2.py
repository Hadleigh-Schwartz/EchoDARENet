"""
UNet with dilated convolutions, nonzero stride to reduce spatial dimensions, 
and only transposed convolutions for decoder layers.
"""

from torch import optim, nn
from torch.optim import lr_scheduler
import torch.nn.functional as F
import pytorch_lightning as pl
import torch as t
import torch.utils.data

import matplotlib.pyplot as plt
import numpy as np

from decoding import CepstralDomainDecodingLoss

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, dilation=1, stride = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size= kernel_size, padding=dilation, dilation=dilation, stride = stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    
class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, dilation=1, stride = 1, output_padding = 0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size= kernel_size, padding = dilation, dilation=dilation, stride = stride, output_padding=output_padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=0.2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetV2(pl.LightningModule):
    """
    UNet with dilated convolutions, nonzero stride to reduce spatial dimensions, 
    and only transpoed convolutions for decoder layers.
    """
    def __init__(self, cfg):
        super().__init__()
        self.name = "UNetV2"

        self.learning_rate = cfg.unet.learning_rate
        self.plot_every_n_steps = cfg.plot_every_n_steps
        self.nwins = cfg.nwins
        self.use_transformer = cfg.unet.use_transformer
        self.residual = cfg.unet.residual
        self.norm_cepstra = cfg.unet.norm_cepstra
        self.alphas = cfg.unet.alphas
        if cfg.unet.cep_target_region is not None:
            self.cepstrum_target_region = cfg.unet.cep_target_region
        else:
            self.cepstrum_target_region = [cfg.Encoding.delays[0] - 10, cfg.Encoding.delays[-1] + 50] # default

        self.lr_scheduler_gamma = 0.9
        self.eps = 1e-16

        #initialize encoding parameters and decoding loss
        self.delays = cfg.Encoding.delays
        self.win_size = cfg.Encoding.win_size
        self.cutoff_freq = cfg.Encoding.cutoff_freq
        self.sample_rate = cfg.sample_rate
        self.decoding_loss = CepstralDomainDecodingLoss(cfg.Encoding.delays, 
                                          cfg.Encoding.win_size, 
                                          cfg.Encoding.cutoff_freq, 
                                          cfg.sample_rate, 
                                          cfg.Encoding.softargmax_beta)

        self.init()

    def init(self, in_channels=2, out_channels=2, base_c=64):
        # Encoder with increasing dilation
        self.enc1 = ConvBlock(in_channels, base_c, dilation=1, stride = 2)
        self.enc2 = ConvBlock(base_c, base_c*2, dilation=2, stride = 2)
        self.enc3 = ConvBlock(base_c*2, base_c*4, dilation=4, stride = 2)
        self.enc4 = ConvBlock(base_c*4, base_c*8, dilation=8, stride = 2)
        self.enc5 = ConvBlock(base_c*8, base_c*8, dilation=8, stride = 2)

        # Decoder
        self.up1 = ConvTransposeBlock(base_c*8, base_c*8, stride=2, output_padding=1)
        self.up2 = ConvTransposeBlock(base_c*16, base_c*4, stride=2, output_padding=1)
        self.up3 = ConvTransposeBlock(base_c*8, base_c*2, stride=2, output_padding=1)
        self.up4 = ConvTransposeBlock(base_c*4, base_c, stride=2, output_padding=1)
        self.up5 = ConvTransposeBlock(base_c*2, out_channels, stride=2, output_padding=1)

        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()


    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        # Decoder
        x = self.up1(x5)
        x = torch.cat((x, x4), dim=1)
        x = self.up2(x)
        x = torch.cat((x, x3), dim=1)
        x = self.up3(x)
        x = torch.cat((x, x2), dim=1)
        x = self.up4(x)
        x = torch.cat((x, x1), dim=1)
        x = self.up5(x)
       
        # Final convolution
        x = self.final_conv(x)
        x = self.tanh(x)

        return x

    def training_step(self, batch, batch_idx):
        loss_type = "train"
        enc_speech_cepstra, enc_reverb_speech_cepstra, unenc_reverb_speech_cepstra, \
                enc_speech_wav, enc_reverb_speech_wav, unenc_reverb_speech_wav, \
                rir, stochastic_noise, noise_condition, symbols,  idx_rir, num_errs_no_reverb, num_errs_reverb = batch
        enc_reverb_speech_cepstra = enc_reverb_speech_cepstra.float()
        enc_speech_cepstra = enc_speech_cepstra.float()
      
        enc_speech_cepstra_hat = self(enc_reverb_speech_cepstra) 

        if batch_idx % self.plot_every_n_steps == 0:
            plot = True
        else:
            plot = False


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
 
        enc_speech_cepstra_hat = self(enc_reverb_speech_cepstra) 

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
        # return optimizer # return here like so if you don't want to use a scheduler
        scheduler = lr_scheduler.ExponentialLR(optimizer, self.lr_scheduler_gamma, self.current_epoch-1)
        return [optimizer], [scheduler]

    