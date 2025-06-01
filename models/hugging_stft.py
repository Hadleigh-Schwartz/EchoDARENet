
from torch import optim, nn
from torch.optim import lr_scheduler
import torch.nn.functional as F
import pytorch_lightning as pl
import torch as t
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, Union

from decoding import CepstralDomainDecodingLoss

from diffusers import UNet2DModel, ConfigMixin, ModelMixin
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D, get_down_block, get_up_block


class HuggingChild(UNet2DModel):
    def __init__(self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 3,
        out_channels: int = 3,
        center_input_sample: bool = False,
        down_block_types: Tuple[str, ...] = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        mid_block_type: Optional[str] = "UNetMidBlock2D",
        up_block_types: Tuple[str, ...] = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        block_out_channels: Tuple[int, ...] = (224, 448, 672, 896),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
        upsample_type: str = "conv",
        dropout: float = 0.0,
        act_fn: str = "silu",
        attention_head_dim: Optional[int] = 8,
        norm_num_groups: int = 32,
        attn_norm_num_groups: Optional[int] = None,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        add_attention: bool = True
        ):
        super().__init__()

        self.sample_size = sample_size
        time_embed_dim = None # my change

        self.center_input_sample = center_input_sample

        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        # input
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))
 
        self.class_embedding = None

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                downsample_padding=downsample_padding,
                resnet_time_scale_shift=resnet_time_scale_shift,
                downsample_type=downsample_type,
                dropout=dropout,
            )
            self.down_blocks.append(down_block)

        # mid
        if mid_block_type is None:
            self.mid_block = None
        else:
            self.mid_block = UNetMidBlock2D(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                dropout=dropout,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else block_out_channels[-1],
                resnet_groups=norm_num_groups,
                attn_groups=attn_norm_num_groups,
                add_attention=add_attention,
            )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                resnet_time_scale_shift=resnet_time_scale_shift,
                upsample_type=upsample_type,
                dropout=dropout,
            )
            self.up_blocks.append(up_block)

        # out
        num_groups_out = norm_num_groups if norm_num_groups is not None else min(block_out_channels[0] // 4, 32)
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=num_groups_out, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)



    def forward(self, sample):
        """
        Overrride the original HuggingFace forward function to remove timestemp.
        Code modified from implementation in diffusers library: https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/models/unets/unet_2d.py#L157
        """
        # 0. center input if necessary
        if self.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time (skipping)

        # 2. pre-process
        skip_sample = sample
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=None, skip_sample=skip_sample # temb is None because we are not using time embeddings/diffusion training
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=None)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(sample, None)

        # 5. up
        skip_sample = None
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, None, skip_sample)
            else:
                sample = upsample_block(sample, res_samples, None)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample

        return sample



class Hugging(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.name = "HuggingFace"

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

        self.model = HuggingChild(
            sample_size=(1536, 32),  # the target image resolution
            in_channels=2,  # the number of input channels, 3 for RGB images
            out_channels=2,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss_type = "train"
        enc_speech_cepstra, enc_reverb_speech_cepstra, unenc_reverb_speech_cepstra, \
                enc_speech_wav, enc_reverb_speech_wav, unenc_reverb_speech_wav, \
                rir, stochastic_noise, noise_condition, symbols,  idx_rir, num_errs_no_reverb, num_errs_reverb = batch
        enc_reverb_speech_cepstra = enc_reverb_speech_cepstra.float() # 2, 1536, 32  (channel, quefrency, num_windows)
        enc_speech_cepstra = enc_speech_cepstra.float() # 2, 1536, 32 (channel, quefrency, num_windows)

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

    