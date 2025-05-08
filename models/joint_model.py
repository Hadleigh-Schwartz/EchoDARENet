from torch import optim, nn
from torch.optim import lr_scheduler
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
from models.fins_lightning_model import FINS
from models.cep4_model import DoubleConv, AttentionBlock
from decoding import CepstralDomainDecodingLoss

class DoubleConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class JointModel(pl.LightningModule):
    def __init__(self, config, in_channels=2, out_channels=2, base_c=64):
        super(JointModel, self).__init__()

        self.config = config
        self.nwins = self.config.nwins
        self.norm_cepstra = self.config.dare.norm_cepstra
        self.alphas = self.config.joint.alphas
        if self.config.dare.cep_target_region is not None:
            self.cepstrum_target_region = self.config.dare.cep_target_region
        else:
            self.cepstrum_target_region = [self.config.Encoding.delays[0] - 10, self.config.Encoding.delays[-1] + 50] # default

        self.lr_scheduler_gamma = 0.9
        self.eps = 1e-16
        self.plot_every_n_steps = self.config.plot_every_n_steps

        #initialize encoding parameters and decoding loss
        self.delays = self.config.Encoding.delays
        self.win_size = self.config.Encoding.win_size
        self.cutoff_freq = self.config.Encoding.cutoff_freq
        self.sample_rate = self.config.sample_rate
        self.decoding_loss = CepstralDomainDecodingLoss(self.delays, 
                                          self.win_size, 
                                          self.cutoff_freq, 
                                          self.sample_rate, 
                                          self.config.dare.softargmax_beta)


        self.fins = FINS(config)

        # Encoder
        self.enc1 = DoubleConv(in_channels, base_c, dilation=1)
        self.enc2 = DoubleConv(base_c, base_c*2, dilation=2)
        self.enc3 = DoubleConv(base_c*2, base_c*4, dilation=4)
        self.enc4 = DoubleConv(base_c*4, base_c*8, dilation=8)
      
        self.rir_enc1 = DoubleConv1D(1, base_c, dilation=1)
        self.rir_enc2 = DoubleConv1D(base_c, base_c*2, dilation=2)
        self.rir_enc3 = DoubleConv1D(base_c*2, base_c*4, dilation=4)
        self.rir_enc4 = DoubleConv1D(base_c*4, base_c*8, dilation=8)
        self.rir_enc5 = DoubleConv1D(base_c*8, base_c*8, dilation=8)

        self.pool = nn.MaxPool2d(2)
        self.pool1d = nn.MaxPool1d(4)

        self.rir_linear = nn.Linear(1, 4)
        z_factor = 2 # how many times the channels of e4 are increased due to concat
        
        # Bottleneck
        self.bottleneck = DoubleConv(base_c*8*z_factor, base_c*16*z_factor)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_c*16*z_factor, base_c*8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_c*16, base_c*8)

        self.up3 = nn.ConvTranspose2d(base_c*8, base_c*4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_c*8, base_c*4)

        self.up2 = nn.ConvTranspose2d(base_c*4, base_c*2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_c*4, base_c*2)

        self.up1 = nn.ConvTranspose2d(base_c*2, base_c, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_c*2, base_c)

        self.final_conv = nn.Conv2d(base_c, out_channels, kernel_size=1)

        
    def forward(self, wav, stochastic_noise, noise_condition,  enc_reverb_speech_cepstra):
        rir_hat = self.fins(wav, stochastic_noise, noise_condition)

        # print if RIR has any NaN or Inf values
        if torch.isnan(rir_hat).any():
            print("RIR has NaN values")
        if torch.isinf(rir_hat).any():
            print("RIR has Inf values")
        # replace NaN and Inf values with 0
        rir_hat[torch.isnan(rir_hat)] = 0
        rir_hat[torch.isinf(rir_hat)] = 0

        # normalize RIR to have max value of 1 to prevent explosins later. Remove??
        rir_hat = rir_hat / (torch.max(rir_hat, dim=2, keepdim=True)[0] + self.eps)

        # pad rir_hat with zeros so it is 49512 samples long
        rir_hat_pad = torch.nn.functional.pad(rir_hat, (0, 49152 - rir_hat.shape[2]), "constant", 0)

        # Cepstra encoder
        e1 = self.enc1(enc_reverb_speech_cepstra)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # RIR encoder
        r_e1 = self.rir_enc1(rir_hat_pad)
        r_e2 = self.rir_enc2(self.pool1d(r_e1))
        r_e3 = self.rir_enc3(self.pool1d(r_e2))
        r_e4 = self.rir_enc4(self.pool1d(r_e3))
        r_e5 = self.rir_enc5(self.pool1d(r_e4))
        r_e5 = r_e5.unsqueeze(3)  # Remove the last dimension
 
        rir_lin = self.rir_linear(r_e5) # lift 1D features to 2D
      
        # contatenate the RIR features with the cepstra features
        z = torch.cat((e4, rir_lin), dim=1)

        # Bottleneck
        b = self.bottleneck(self.pool(z))
     
        # Decoder
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return rir_hat, self.final_conv(d1)
 
    def training_step(self, batch, batch_idx):
        loss_type = "train"

        enc_speech_cepstra, enc_reverb_speech_cepstra, unenc_reverb_speech_cepstra, \
        enc_speech_wav, enc_reverb_speech_wav, unenc_reverb_speech_wav, \
        rir, stochastic_noise, noise_condition, symbols,  idx_rir, num_errs_no_reverb, num_errs_reverb  = batch
        # Q: Should we use the same noise for both RIR predictions???
 
        # Convert speech wavs and noise to floats
        enc_reverb_speech_wav = enc_reverb_speech_wav.float()
        unenc_reverb_speech_wav = unenc_reverb_speech_wav.float()
        enc_reverb_speech_cepstra = enc_reverb_speech_cepstra.float()
        unenc_reverb_speech_cepstra = unenc_reverb_speech_cepstra.float()
        enc_speech_cepstra = enc_speech_cepstra.float()
        rir = rir.float()
        stochastic_noise = stochastic_noise.float()
        noise_condition = noise_condition.float()

        # Get predicted RIR approximates
        enc_rir_hat, enc_speech_cepstra_hat = self(enc_reverb_speech_wav, stochastic_noise, noise_condition, enc_reverb_speech_cepstra)
        enc_rir_hat = enc_rir_hat.squeeze(1) 
        unenc_rir_hat, _ = self(unenc_reverb_speech_wav, stochastic_noise, noise_condition, unenc_reverb_speech_cepstra) # unenc cepstra just there for consistency
        unenc_rir_hat = unenc_rir_hat.squeeze(1)
        
        # (Optional) Compute FINS RIR loss for enc
        enc_stft_loss_dict = self.fins.stft_loss_fn(enc_rir_hat, rir)
        enc_stft_loss = enc_stft_loss_dict["total"]
        unenc_stft_loss_dict = self.fins.stft_loss_fn(unenc_rir_hat, rir)
        unenc_stft_loss = unenc_stft_loss_dict["total"]
        sum_stft_loss = enc_stft_loss + unenc_stft_loss

        # Compute MSE of the enc and unenc RIRs
        rir_mse = nn.functional.mse_loss(enc_rir_hat, unenc_rir_hat)

        # Cepstral loss
        cepstra_loss, full_cepstra_loss = self.compute_speech_loss(enc_speech_cepstra, enc_speech_cepstra_hat ) 
        sym_err_rate, avg_err_reduction_loss, no_reverb_sym_err_rate, reverb_sym_err_rate  = self.decoding_loss(enc_speech_cepstra_hat, symbols, num_errs_no_reverb, num_errs_reverb)
        
        # Final loss
        loss = self.alphas[0] * cepstra_loss + self.alphas[1] * sym_err_rate + sum_stft_loss + rir_mse * 1000

        if batch_idx % self.plot_every_n_steps == 0:
            self.plot_rirs(rir, enc_rir_hat, batch_idx, loss_type=loss_type)
            self.make_cepstra_plot(enc_speech_cepstra, enc_speech_cepstra, enc_speech_cepstra_hat, symbols, loss_type=loss_type)

        self.log(loss_type+"_enc_stft_loss" + loss_type, enc_stft_loss, sync_dist = True )
        self.log(loss_type+"_unenc_stft_loss" + loss_type, unenc_stft_loss, sync_dist = True )
        self.log(loss_type+"_sum_stft_loss", sum_stft_loss, sync_dist = True )
        self.log(loss_type+"_rir_mse" , rir_mse, sync_dist = True )
        self.log(loss_type+"_cep_mse_loss", cepstra_loss, sync_dist = True )
        self.log(loss_type+"_sym_err_rate", sym_err_rate, sync_dist = True )
        self.log(loss_type+"_full_cep_mse_loss", full_cepstra_loss, sync_dist = True )
        self.log(loss_type+"_avg_err_reduction_loss", avg_err_reduction_loss, sync_dist = True )
        self.log(loss_type+"_no_reverb_sym_err_rate", no_reverb_sym_err_rate, sync_dist = True )
        self.log(loss_type+"_reverb_sym_err_rate", reverb_sym_err_rate, sync_dist = True )
        self.log(loss_type+"_loss", loss, sync_dist = True )
        
        return loss


    def validation_step(self, batch, batch_idx):
        loss_type = "val"

        enc_speech_cepstra, enc_reverb_speech_cepstra, unenc_reverb_speech_cepstra, \
        enc_speech_wav, enc_reverb_speech_wav, unenc_reverb_speech_wav, \
        rir, stochastic_noise, noise_condition, symbols,  idx_rir, num_errs_no_reverb, num_errs_reverb  = batch
        # Q: Should we use the same noise for both RIR predictions???
 
        # Convert speech wavs and noise to floats
        enc_reverb_speech_wav = enc_reverb_speech_wav.float()
        unenc_reverb_speech_wav = unenc_reverb_speech_wav.float()
        enc_reverb_speech_cepstra = enc_reverb_speech_cepstra.float()
        unenc_reverb_speech_cepstra = unenc_reverb_speech_cepstra.float()
        enc_speech_cepstra = enc_speech_cepstra.float()
        rir = rir.float()
        stochastic_noise = stochastic_noise.float()
        noise_condition = noise_condition.float()

        # Get predicted RIR approximates
        enc_rir_hat, enc_speech_cepstra_hat = self(enc_reverb_speech_wav, stochastic_noise, noise_condition, enc_reverb_speech_cepstra)
        enc_rir_hat = enc_rir_hat.squeeze(1)
        unenc_rir_hat, _ = self(unenc_reverb_speech_wav, stochastic_noise, noise_condition, unenc_reverb_speech_cepstra)
        unenc_rir_hat = unenc_rir_hat.squeeze(1)

        # (Optional) Compute FINS RIR loss for enc
        enc_stft_loss_dict = self.fins.stft_loss_fn(enc_rir_hat, rir)
        enc_stft_loss = enc_stft_loss_dict["total"]
        unenc_stft_loss_dict = self.fins.stft_loss_fn(unenc_rir_hat, rir)
        unenc_stft_loss = unenc_stft_loss_dict["total"]
        sum_stft_loss = enc_stft_loss + unenc_stft_loss

        # Compute MSE of the enc and unenc RIRs
        rir_mse = nn.functional.mse_loss(enc_rir_hat, unenc_rir_hat)
        
        # Cepstral loss
        cepstra_loss, full_cepstra_loss = self.compute_speech_loss(enc_speech_cepstra, enc_speech_cepstra_hat ) 
        sym_err_rate, avg_err_reduction_loss, no_reverb_sym_err_rate, reverb_sym_err_rate  = self.decoding_loss(enc_speech_cepstra_hat, symbols, num_errs_no_reverb, num_errs_reverb)
        
        # Final loss
        loss = self.alphas[0] * cepstra_loss + self.alphas[1] * sym_err_rate + sum_stft_loss + rir_mse * 1000

        self.plot_rirs(rir, enc_rir_hat, batch_idx, loss_type=loss_type)
        self.make_cepstra_plot(enc_speech_cepstra, enc_speech_cepstra, enc_speech_cepstra_hat, symbols, loss_type=loss_type)
        
        self.log(loss_type+"_enc_stft_loss" + loss_type, enc_stft_loss, sync_dist = True )
        self.log(loss_type+"_unenc_stft_loss" + loss_type, unenc_stft_loss, sync_dist = True )
        self.log(loss_type+"_sum_stft_loss", sum_stft_loss, sync_dist = True )
        self.log(loss_type+"_rir_mse" , rir_mse, sync_dist = True )
        self.log(loss_type+"_cep_mse_loss", cepstra_loss, sync_dist = True )
        self.log(loss_type+"_sym_err_rate", sym_err_rate, sync_dist = True )
        self.log(loss_type+"_full_cep_mse_loss", full_cepstra_loss, sync_dist = True )
        self.log(loss_type+"_avg_err_reduction_loss", avg_err_reduction_loss, sync_dist = True )
        self.log(loss_type+"_no_reverb_sym_err_rate", no_reverb_sym_err_rate, sync_dist = True )
        self.log(loss_type+"_reverb_sym_err_rate", reverb_sym_err_rate, sync_dist = True )
        self.log(loss_type+"_loss", loss, sync_dist = True )
        
        return loss
    
    def test_step(self, batch, batch_idx):
       pass


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
  
    
    def plot_rirs(self, gt_rir, predicted_rir, batch_idx, loss_type="train"):
        fh = plt.figure()
        fig, axes = plt.subplots(2, 2, figsize=(12, 5), tight_layout=True)
        batch_size = gt_rir.shape[0]
        batch_els = np.random.choice(batch_size, 4, replace=False)
        for i, batch_el in enumerate(batch_els):
            axes[i // 2, i % 2].plot(gt_rir[batch_el].detach().cpu().numpy()[:10000], label="GT RIR", alpha = 0.5)
            axes[i // 2, i % 2].plot(predicted_rir[batch_el].detach().cpu().numpy()[:10000], label="Predicted RIR", alpha = 0.5)
            axes[i // 2, i % 2].set_title(f"Batch element {batch_el}")
            axes[i // 2, i % 2].legend()
        plt.suptitle(f"Batch {batch_idx} - {loss_type} RIRs")
       
        tb = self.logger.experiment
        tb.add_figure('RIR_'+loss_type, fig, global_step=self.global_step)
        plt.close()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config.joint.lr)
        # return optimizer
        scheduler = lr_scheduler.ExponentialLR(optimizer, self.lr_scheduler_gamma, self.current_epoch-1)
        return [optimizer], [scheduler]
