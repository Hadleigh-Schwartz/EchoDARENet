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
        self.name = "EchoSpeechDAREUnet"

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

    def init(self):
        k = 5
        s = 2
        p_drop = 0.5
        leaky_slope = 0.2
        
        self.conv1 = nn.Sequential(nn.Conv2d(  2,  64, k, stride=s, padding=k//2), nn.LeakyReLU(leaky_slope))
        self.conv2 = nn.Sequential(nn.Conv2d( 64, 128, k, stride=s, padding=k//2), nn.BatchNorm2d(128), nn.LeakyReLU(leaky_slope))
        self.conv3 = nn.Sequential(nn.Conv2d( 128, 256, k, stride=s, padding=k//2), nn.BatchNorm2d(256), nn.LeakyReLU(leaky_slope))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, k, stride=s, padding=k//2), nn.BatchNorm2d(512), nn.LeakyReLU(leaky_slope))
        self.conv5 = nn.Sequential(nn.Conv2d(512, 512, k, stride=s, padding=k//2), nn.BatchNorm2d(512), nn.ReLU())
        
        if self.use_transformer:
            self.transformer = nn.Transformer(d_model=512, nhead=4, num_encoder_layers=5, num_decoder_layers=5, batch_first=True)

        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(512, 512, k, stride=s, padding=k//2, output_padding=s-1), nn.BatchNorm2d(512), nn.Dropout2d(p=p_drop), nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(1024, 256, k, stride=s, padding=k//2, output_padding=s-1), nn.BatchNorm2d(256), nn.Dropout2d(p=p_drop), nn.ReLU())
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(512,  128, k, stride=s, padding=k//2, output_padding=s-1), nn.BatchNorm2d(128),  nn.Dropout2d(p=p_drop), nn.ReLU())
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(256,   64, k, stride=s, padding=k//2, output_padding=s-1), nn.BatchNorm2d(64), nn.ReLU()) # important to have Tanh not previous version's ReLU otherwise can't be negative
        self.deconv5 = nn.Sequential(nn.ConvTranspose2d(128,   2, k, stride=s, padding=k//2, output_padding=s-1), nn.Tanh()) 

    def predict(self, x):
        if self.residual:
            residual = x

        c1Out = self.conv1(x)     
        c2Out = self.conv2(c1Out) 
        c3Out = self.conv3(c2Out) 
        c4Out = self.conv4(c3Out) 
        c5Out = self.conv5(c4Out) 
    
        if self.use_transformer:
            c5Out = self.transformer(\
                c5Out.squeeze().permute((0,2,1)), \
                c5Out.squeeze().permute((0,2,1))).permute((0,2,1)).unsqueeze(-1)
        
        d1Out = self.deconv1(c5Out) 
        d2Out = self.deconv2(t.cat((d1Out, c4Out), dim=1)) 
        d3Out = self.deconv3(t.cat((d2Out, c3Out), dim=1)) 
        d4Out = self.deconv4(t.cat((d3Out, c2Out), dim=1)) 
        d5Out = self.deconv5(t.cat((d4Out, c1Out), dim=1)) 

        if self.residual:
            cep_out = d5Out + residual
        else:
            cep_out = d5Out

        return cep_out

    def training_step(self, batch, batch_idx):
        loss_type = "train"
        enc_speech_cepstra, enc_reverb_speech_cepstra, unenc_reverb_speech_cepstra, \
                enc_speech_wav, enc_reverb_speech_wav, unenc_reverb_speech_wav, \
                rir, stochastic_noise, noise_condition, symbols,  idx_rir = batch
        enc_reverb_speech_cepstra = enc_reverb_speech_cepstra.float()
        enc_speech_cepstra = enc_speech_cepstra.float()
        num_errs_no_reverb = torch.tensor(0) # DUMMY FOR NOW
        num_errs_reverb = torch.tensor(0) # DUMMY FOR NOW

        # x, ys, zs, ys_t, y, yt, _ , symbols, num_errs_no_reverb, num_errs_reverb, idxs_rir = batch # reverberant speech cepstra, clean speech cepstra, RIR fft, RIR time domain, symbols echo-encoded into speech, error rate of echo-decoding pre-reverb
        # ys = ys.float()
        # zs = zs.float()
        # yt = yt.float()
        # x = x.float()
        # y = y.float()
        enc_speech_cepstra_hat = self.predict(enc_reverb_speech_cepstra) 

        if batch_idx % self.plot_every_n_steps == 0:
            plot = True
        else:
            plot = False


        cepstra_loss = self.compute_speech_loss(enc_speech_cepstra, enc_speech_cepstra_hat ) 
        sym_err_rate, avg_err_reduction_loss, no_reverb_sym_err_rate, reverb_sym_err_rate  = self.decoding_loss(enc_speech_cepstra_hat, symbols, num_errs_no_reverb, num_errs_reverb)
        loss = self.alphas[0] * cepstra_loss + self.alphas[1] * sym_err_rate + self.alphas[2] * avg_err_reduction_loss 
        
        self.log(loss_type+"_cep_mse_loss", cepstra_loss, sync_dist = True )
        self.log(loss_type+"_sym_err_rate", sym_err_rate, sync_dist = True )
        self.log(loss_type+"_avg_err_reduction_loss", avg_err_reduction_loss, sync_dist = True )
        self.log(loss_type+"_no_reverb_sym_err_rate", no_reverb_sym_err_rate, sync_dist = True )
        self.log(loss_type+"_reverb_sym_err_rate", reverb_sym_err_rate, sync_dist = True )
        self.log(loss_type+"_loss", loss, sync_dist = True )
      
        if plot:
            self.make_cepstra_plot(enc_reverb_speech_cepstra, enc_speech_cepstra, enc_speech_cepstra_hat, symbols)
            self.weight_histograms()
            
        return loss

    def validation_step(self, batch, batch_idx):
        loss_type = "val"

        enc_speech_cepstra, enc_reverb_speech_cepstra, unenc_reverb_speech_cepstra, \
                enc_speech_wav, enc_reverb_speech_wav, unenc_reverb_speech_wav, \
                rir, stochastic_noise, noise_condition, symbols,  idx_rir = batch
        enc_reverb_speech_cepstra = enc_reverb_speech_cepstra.float()
        enc_speech_cepstra = enc_speech_cepstra.float()
        num_errs_no_reverb = torch.tensor(0) # DUMMY FOR NOW
        num_errs_reverb = torch.tensor(0) # DUMMY FOR NOW

        enc_speech_cepstra_hat = self.predict(enc_reverb_speech_cepstra) 

        cepstra_loss = self.compute_speech_loss(enc_speech_cepstra, enc_speech_cepstra_hat ) 
        sym_err_rate, avg_err_reduction_loss, no_reverb_sym_err_rate, reverb_sym_err_rate  = self.decoding_loss(enc_speech_cepstra_hat, symbols, num_errs_no_reverb, num_errs_reverb)
        loss = self.alphas[0] * cepstra_loss + self.alphas[1] * sym_err_rate + self.alphas[2] * avg_err_reduction_loss 
        
        self.log(loss_type+"_cep_mse_loss", cepstra_loss, sync_dist = True )
        self.log(loss_type+"_sym_err_rate", sym_err_rate, sync_dist = True )
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
        return mse_abs
    

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
            layers.append(self.conv5[0])
            if self.use_transformer:
                layers.append(self.transformer)
            layers.append(self.deconv1[0])
            layers.append(self.deconv2[0])
            layers.append(self.deconv3[0])
            layers.append(self.deconv4[0])
            layers.append(self.deconv5[0])
            
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

    