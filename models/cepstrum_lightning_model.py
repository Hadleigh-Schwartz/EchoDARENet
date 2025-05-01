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
        leaky_slope = 0.01
        
        self.conv1 = nn.Sequential(nn.Conv2d(  2,  64, k, stride=s, padding=k//2), nn.LeakyReLU(leaky_slope))
        self.conv2 = nn.Sequential(nn.Conv2d( 64, 128, k, stride=s, padding=k//2), nn.BatchNorm2d(128), nn.LeakyReLU(leaky_slope))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, k, stride=s, padding=k//2), nn.BatchNorm2d(256), nn.LeakyReLU(leaky_slope))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 256, k, stride=s, padding=k//2), nn.BatchNorm2d(256), nn.ReLU())
        
        if self.use_transformer:
            self.transformer = nn.Transformer(d_model=256, nhead=4, num_encoder_layers=5, num_decoder_layers=5, batch_first=True)

        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(256, 256, k, stride=s, padding=k//2, output_padding=s-1), nn.BatchNorm2d(256), nn.Dropout2d(p=p_drop), nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(512, 128, k, stride=s, padding=k//2, output_padding=s-1), nn.BatchNorm2d(128), nn.Dropout2d(p=p_drop), nn.ReLU())
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(256,  64, k, stride=s, padding=k//2, output_padding=s-1), nn.BatchNorm2d(64),  nn.ReLU())
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(128,   2, k, stride=s, padding=k//2, output_padding=s-1), nn.BatchNorm2d(2),  nn.Tanh()) # important to have Tanh not previous version's ReLU otherwise can't be negative

    
        # num_rirs = 40
        # self.disc_deconv1 = nn.Sequential(nn.ConvTranspose2d(256, 256, k, stride=s, padding=k//2, output_padding=s-1), nn.BatchNorm2d(256), nn.Dropout2d(p=p_drop), nn.ReLU())
        # self.disc_deconv2 = nn.Sequential(nn.ConvTranspose2d(256, 128, k, stride=s, padding=k//2, output_padding=s-1), nn.BatchNorm2d(128), nn.Dropout2d(p=p_drop), nn.ReLU())
        # self.disc_deconv3 = nn.Sequential(nn.ConvTranspose2d(128,  64, k, stride=s, padding=k//2, output_padding=s-1), nn.BatchNorm2d(64),  nn.ReLU())
        # self.disc_deconv4 = nn.Sequential(nn.ConvTranspose2d(64,   1, k, stride=s, padding=k//2, output_padding=s-1), nn.BatchNorm2d(1),  nn.Tanh()) # important to have Tanh not previous version's ReLU otherwise can't be negative
        # # self.disc_conv1 = nn.Sequential(nn.ConvTranspose2d(256, 256, k, stride=s, padding=k//2, output_padding=s-1), nn.BatchNorm2d(256), nn.Dropout2d(p=p_drop), nn.ReLU())
        # # self.disc_conv2 = nn.Sequential(nn.Conv2d( 512, 128, k, stride=s, padding=k//2), nn.BatchNorm2d(128), nn.LeakyReLU(leaky_slope))
        # # self.disc_conv3 = nn.Sequential(nn.Conv2d(256, 64, k, stride=s, padding=k//2), nn.BatchNorm2d(64), nn.LeakyReLU(leaky_slope))
        # # self.disc_conv4 = nn.Sequential(nn.Conv2d(128, 2, k, stride=s, padding=k//2), nn.BatchNorm2d(2), nn.ReLU())
        # # inspired by ArcFace https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/backbones/iresnet.py
        # # flatten deconv5 output, then proceed
        # self.dropout = nn.Dropout(p=0.25)
        # self.fc = nn.Linear(16384, num_rirs) #65536=2048*32
        # self.features = nn.BatchNorm1d(num_rirs, eps=1e-05)
        # # nn.init.constant_(self.features.weight, 1.0)
        # # self.features.weight.requires_grad = False

    def predict(self, x):
        if self.residual:
            residual = x

        c1Out = self.conv1(x)     # (64 x 64 x  64)
        c2Out = self.conv2(c1Out) # (16 x 16 x 128)
        c3Out = self.conv3(c2Out) # ( 4 x  4 x 256)
        c4Out = self.conv4(c3Out) # ( 1 x  1 x 256)

        if self.use_transformer:
            c4Out = self.transformer(\
                c4Out.squeeze().permute((0,2,1)), \
                c4Out.squeeze().permute((0,2,1))).permute((0,2,1)).unsqueeze(-1)
        
        d1Out = self.deconv1(c4Out) # (  4 x   4 x 256)
        d2Out = self.deconv2(t.cat((d1Out, c3Out), dim=1)) # ( 16 x  16 x 128)
        d3Out = self.deconv3(t.cat((d2Out, c2Out), dim=1)) # ( 64 x  64 x 128)
        d4Out = self.deconv4(t.cat((d3Out, c1Out), dim=1)) # (256 x 256 x 1)

        if self.residual:
            cep_out = d4Out + residual
        else:
            cep_out = d4Out

    
        # #compute the value of alpha
        # if epoch >0 and total>0:
        #     p = float(1 + (epoch) * total) / (epoch) / total
        #     alpha = 2. / (1. + np.exp(-10 * p)) - 1
        # else:
        #     alpha = 0.5

        # # rir representation branch
        # print("c4Out shape: ", c4Out.shape)
        # discOut = self.disc_deconv1(c4Out)
        # discOut = self.disc_deconv2(discOut)
        # discOut = self.disc_deconv3(discOut)
        # discOut = self.disc_deconv4(discOut)
        # # print("post deconv discOut shape: ", discOut.shape)
        # # discOut = self.disc_conv1(discOut)    
        # # discOut = self.disc_conv2(discOut)
        # # discOut = self.disc_conv3(discOut)
        # # discOut = self.disc_conv4(discOut)
        # discOut = torch.flatten(discOut, 1)  # flatten the output for the fully connected layer
        # # print("post conv discOut shape: ", discOut.shape)
        # discOut = self.dropout(discOut)  # apply dropout
        # # print( "After dropout, discOut shape: ", discOut.shape)
        # discOut = self.fc(discOut)  # fully connected layer
        # # print("After fc, discOut shape: ", discOut.shape)
        # discOut = self.features(discOut)  # batch normalization
        # # print( "After features, discOut shape: ", discOut.shape)
        discOut = None
  
        return cep_out, discOut, c4Out

    def training_step(self, batch, batch_idx):
        loss_type = "train"

        x, ys, zs, ys_t, y, yt, _ , symbols, num_errs_no_reverb, num_errs_reverb, idxs_rir = batch # reverberant speech cepstra, clean speech cepstra, RIR fft, RIR time domain, symbols echo-encoded into speech, error rate of echo-decoding pre-reverb
        ys = ys.float()
        zs = zs.float()
        yt = yt.float()
        x = x.float()
        y = y.float()

        ys_hat, _, bottleneck = self.predict(x) 
        # _, _, bottleneck2 = self.predict(zs)

      
        if batch_idx % self.plot_every_n_steps == 0:
            plot = True
        else:
            plot = False

        # pair_loss = nn.functional.mse_loss(bottleneck, bottleneck2)
        pair_loss = 0
        cepstra_loss = self.compute_speech_loss(ys, ys_hat) # the speech prediction branch
        sym_err_rate, avg_err_reduction_loss, no_reverb_sym_err_rate, reverb_sym_err_rate  = self.decoding_loss(ys_hat, symbols, num_errs_no_reverb, num_errs_reverb)
        loss = self.alphas[0] * cepstra_loss + self.alphas[1] * sym_err_rate + self.alphas[2] * avg_err_reduction_loss + self.alphas[3] * pair_loss
        
        self.log(loss_type+"_cep_mse_loss", cepstra_loss, sync_dist = True )
        self.log(loss_type+"_sym_err_rate", sym_err_rate, sync_dist = True )
        self.log(loss_type+"_avg_err_reduction_loss", avg_err_reduction_loss, sync_dist = True )
        self.log(loss_type+"_no_reverb_sym_err_rate", no_reverb_sym_err_rate, sync_dist = True )
        self.log(loss_type+"_reverb_sym_err_rate", reverb_sym_err_rate, sync_dist = True )
        self.log(loss_type+"_pair_loss", pair_loss, sync_dist = True )
        self.log(loss_type+"_loss", loss, sync_dist = True )
      
        if plot:
            self.make_cepstra_plot(x, ys, ys_hat, symbols)
            
        return loss

    def validation_step(self, batch, batch_idx):
        loss_type = "val"
        x, ys, zs, ys_t, y, yt, _ , symbols, num_errs_no_reverb, num_errs_reverb, idxs_rir = batch # reverberant speech cepstra, clean speech cepstra, RIR fft, RIR time domain, symbols echo-encoded into speech, error rate of echo-decoding pre-reverb
        ys = ys.float()
        zs = zs.float()
        yt = yt.float()
        x = x.float()
        y = y.float()

        ys_hat, _, bottleneck = self.predict(x) 
        # _, _, bottleneck2 = self.predict(zs)
        # pair_loss = nn.functional.mse_loss(bottleneck, bottleneck2)
        pair_loss = 0
        cepstra_loss = self.compute_speech_loss(ys, ys_hat) # the speech prediction branch
        sym_err_rate, avg_err_reduction_loss, no_reverb_sym_err_rate, reverb_sym_err_rate  = self.decoding_loss(ys_hat, symbols, num_errs_no_reverb, num_errs_reverb)
        loss = self.alphas[0] * cepstra_loss + self.alphas[1] * sym_err_rate + self.alphas[2] * avg_err_reduction_loss + self.alphas[3] * pair_loss
        
        self.log(loss_type+"_cep_mse_loss", cepstra_loss, sync_dist = True )
        self.log(loss_type+"_sym_err_rate", sym_err_rate, sync_dist = True )
        self.log(loss_type+"_avg_err_reduction_loss", avg_err_reduction_loss, sync_dist = True )
        self.log(loss_type+"_no_reverb_sym_err_rate", no_reverb_sym_err_rate, sync_dist = True )
        self.log(loss_type+"_reverb_sym_err_rate", reverb_sym_err_rate, sync_dist = True )
        self.log(loss_type+"_pair_loss", pair_loss, sync_dist = True )
        self.log(loss_type+"_loss", loss, sync_dist = True )
      
      
        self.make_cepstra_plot(x, ys, ys_hat, symbols)
        
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
    
    def compute_intrabatch_rir_mse(self, y_hat, loss_type = "train"):
        # https://discuss.pytorch.org/t/how-to-calculate-pair-wise-differences-between-two-tensors-in-a-vectorized-way/37451
        tensor_a = y_hat.unsqueeze(0)
        tensor_b = y_hat.unsqueeze(1)
        squared_diff = (tensor_a - tensor_b) ** 2
        squared_diff = torch.mean(squared_diff, dim=2)
        # print(squared_diff[1, 0], torch.mean((y_hat[0, :] - y_hat[1, :]) ** 2))
        # print(squared_diff[0, 1], torch.mean((y_hat[1, :] - y_hat[0, :]) ** 2))
        # print(squared_diff[2, 3], torch.mean((y_hat[2, :] - y_hat[3, :]) ** 2))
        # print(squared_diff[3, 2], torch.mean((y_hat[2, :] - y_hat[3, :]) ** 2))
        squared_diff = squared_diff.triu(diagonal=1)         # zero out half of the matrix as it is symmetric
        N = math.comb(y_hat.shape[0], 2)
        intrabatch_rir_mse = torch.sum(squared_diff) /  N
        zero_penalty = (y_hat * (1.0 - y_hat))**2
        zero_penalty = torch.mean(zero_penalty)
        intrabatch_rir_mse = intrabatch_rir_mse + zero_penalty
        return intrabatch_rir_mse
        
    def compute_rir_loss(self, y, yt, y_predict, plot = False, loss_type = "train"):
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
            plot: bool
                whether to plot the RIR
            loss_type: str
                "train", "val", or "test"
            
        Returns:
            All of the computed losses
        """
        n_rir_gt = y.shape[2]
        n_rir_pred = y_predict.shape[2]
        y_c = y[:,0,::n_rir_gt//n_rir_pred,:].float() + 1j*y[:,1,::n_rir_gt//n_rir_pred,:].float()
        
        y_hat_c = y_predict[:,0,:,:] + 1j*y_predict[:,1,:,:]
        y_hat_c_t = t.fft.irfft(y_hat_c,n=yt.shape[1], dim=1).squeeze() # reverse predicted frequency-domain representation to time-domain
 
        target_rir_portion = y_hat_c_t[:,self.delays[-1] + 10:]
        # https://discuss.pytorch.org/t/how-to-calculate-pair-wise-differences-between-two-tensors-in-a-vectorized-way/37451
        tensor_a = target_rir_portion.unsqueeze(0)
        tensor_b = target_rir_portion.unsqueeze(1)
        squared_diff = (tensor_a - tensor_b) ** 2
        squared_diff = torch.mean(squared_diff, dim=2)
        squared_diff = squared_diff.triu(diagonal=1)         # take half of the matrix as the other half is symmetric
        N = math.comb(target_rir_portion.shape[0], 2)
        intrabatch_rir_mse = torch.sum(squared_diff) /  N
        # print(squared_diff[1, 0], torch.mean((target_rir_portion[0, :] - target_rir_portion[1, :]) ** 2))
        # print(squared_diff[0, 1], torch.mean((target_rir_portion[1, :] - target_rir_portion[0, :]) ** 2))
        # print(squared_diff[2, 3], torch.mean((target_rir_portion[2, :] - target_rir_portion[3, :]) ** 2))
        # print(squared_diff[3, 2], torch.mean((target_rir_portion[2, :] - target_rir_portion[3, :]) ** 2))

        mse_time = nn.functional.mse_loss(y_hat_c_t[:,:800], yt[:,:800]) # TODO: make this in config
        err_time = nn.functional.l1_loss(y_hat_c_t, yt)
        y_hat_c_t_abs_log = (y_hat_c_t.abs()+self.eps).log()
        yt_abs_log = (yt.abs()+self.eps).log()
        mse_time_abs_log = nn.functional.mse_loss(y_hat_c_t_abs_log, yt_abs_log)
        err_time_abs_log = nn.functional.l1_loss(y_hat_c_t_abs_log, yt_abs_log)
        kld_time_abs_log = nn.functional.kl_div(y_hat_c_t_abs_log,yt_abs_log,log_target=True).abs()

        err_timedelay = t.mean(t.log(t.abs(t.argmax(y_hat_c_t,dim=1) - t.argmax(yt))+1))
        err_peak = 0.5*t.mean(t.abs(t.argmax(y_hat_c_t, dim = 1) - t.argmax(yt))/yt.shape[1] \
            + 0.5*t.abs(t.max(y_hat_c_t,dim=1)[0] - t.max(yt)))
        err_peakval = t.mean(t.abs(y_hat_c_t[:,t.argmax(yt)] - t.max(yt)))
    
        mse_real = nn.functional.mse_loss(t.real(y_hat_c),t.real(y_c))
        mse_imag = nn.functional.mse_loss(t.imag(y_hat_c),t.imag(y_c))
        mse_abs = nn.functional.mse_loss(t.log(t.abs(y_hat_c)),t.log(t.abs(y_c)))
        
        err_real = nn.functional.l1_loss(t.real(y_hat_c),t.real(y_c))
        err_imag = nn.functional.l1_loss(t.imag(y_hat_c),t.imag(y_c))
        err_abs = nn.functional.l1_loss(t.log(t.abs(y_hat_c)),t.log(t.abs(y_c)))
     
        y1 = t.sin(t.angle(y_c))
        y2 = t.cos(t.angle(y_c))
        y_hat1 = t.sin(t.angle(y_hat_c))
        y_hat2 = t.cos(t.angle(y_hat_c))
        
        mse_phase = nn.functional.mse_loss(y1,y_hat1) + nn.functional.mse_loss(y2,y_hat2)
        err_phase = nn.functional.l1_loss(y1,y_hat1) + nn.functional.l1_loss(y2,y_hat2)
        y_a = t.tensor(np.unwrap(t.angle(y_c).cpu().detach().numpy(),axis=1)).to(self.device)
        y_hat_a = t.tensor(np.unwrap(t.angle(y_hat_c).cpu().detach().numpy(),axis=1)).to(self.device)
        mse_phase_un = nn.functional.mse_loss(y_a,y_hat_a)
        err_phase_un = nn.functional.l1_loss(y_a,y_hat_a)

        #loss = err_real + err_imag + 2*err_abs
        loss_err = err_abs + err_phase # + err_peakval # + err_timedelay #+ err_phase_un * 1e-4
        loss_mse = mse_abs + mse_phase # + err_peakval # + err_timedelay #+ mse_phase_un * 1e-4

        if plot == True:
            self.make_rir_plot(yt, y_hat_c_t, loss_type, intrabatch_rir_mse = intrabatch_rir_mse )
        
        return \
            loss_err, loss_mse, \
            err_real, err_imag, err_abs, None, err_phase, err_phase_un, err_time, err_time_abs_log, \
            mse_real, mse_imag, mse_abs, None, mse_phase, mse_phase_un, mse_time, mse_time_abs_log, \
            kld_time_abs_log, err_timedelay, err_peak, err_peakval, \
            y_hat_c, intrabatch_rir_mse
    

    def make_cepstra_plot(self, input_cepstra, clean_cepstra, pred_cepstra, symbols, loss_type = "train"):
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

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        # return optimizer
        scheduler = lr_scheduler.ExponentialLR(optimizer, self.lr_scheduler_gamma, self.current_epoch-1)
        return [optimizer], [scheduler]

    