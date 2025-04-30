from torch import optim, nn
from torch.optim import lr_scheduler
import pytorch_lightning as pl
import torch as t
import torch.utils.data
import torchaudio as ta
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.utils import spectral_norm

from fins_loss import MultiResolutionSTFTLoss
from fins.fins.utils.audio import (
    get_octave_filters,
)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super(EncoderBlock, self).__init__()
        if use_batchnorm:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=15, stride=2, padding=7),
                nn.BatchNorm1d(out_channels, track_running_stats=True),
                nn.PReLU(),
            )
            self.skip_conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2, padding=0),
                nn.BatchNorm1d(out_channels, track_running_stats=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=15, stride=2, padding=7),
                nn.PReLU(),
            )
            self.skip_conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2, padding=0),
            )

    def forward(self, x):
        out = self.conv(x)
        skip_out = self.skip_conv(x)
        skip_out = out + skip_out
        return skip_out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        block_list = []
        channels = [1, 32, 32, 64, 64, 64, 128, 128, 128, 256, 256, 256, 512, 512]

        for i in range(0, len(channels) - 1):
            if (i + 1) % 3 == 0:
                use_batchnorm = True
            else:
                use_batchnorm = False
            in_channels = channels[i]
            out_channels = channels[i + 1]
            curr_block = EncoderBlock(in_channels, out_channels, use_batchnorm)
            block_list.append(curr_block)

        self.encode = nn.Sequential(*block_list)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, 128)

    def forward(self, x):
        b, c, l = x.size()
        out = self.encode(x)
        out = self.pooling(out)
        out = out.view(b, -1)
        out = self.fc(out)
        return out


class UpsampleNet(nn.Module):
    def __init__(self, input_size, output_size, upsample_factor):
        super(UpsampleNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.upsample_factor = upsample_factor

        layer = nn.ConvTranspose1d(
            input_size,
            output_size,
            upsample_factor * 2,
            upsample_factor,
            padding=upsample_factor // 2,
        )
        nn.init.orthogonal_(layer.weight)
        self.layer = spectral_norm(layer)

    def forward(self, inputs):
        outputs = self.layer(inputs)
        outputs = outputs[:, :, : inputs.size(-1) * self.upsample_factor]
        return outputs


class ConditionalBatchNorm1d(nn.Module):

    """Conditional Batch Normalization"""

    def __init__(self, num_features, condition_length):
        super().__init__()

        self.num_features = num_features
        self.condition_length = condition_length
        self.norm = nn.BatchNorm1d(num_features, affine=True, track_running_stats=True)

        self.layer = spectral_norm(nn.Linear(condition_length, num_features * 2))
        self.layer.weight.data.normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.layer.bias.data.zero_()  # Initialise bias at 0

    def forward(self, inputs, noise):
        outputs = self.norm(inputs)
        gamma, beta = self.layer(noise).chunk(2, 1)
        gamma = gamma.view(-1, self.num_features, 1)
        beta = beta.view(-1, self.num_features, 1)

        outputs = gamma * outputs + beta

        return outputs


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_factor, condition_length):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.condition_length = condition_length
        self.upsample_factor = upsample_factor

        # Block A
        self.condition_batchnorm1 = ConditionalBatchNorm1d(in_channels, condition_length)

        self.first_stack = nn.Sequential(
            nn.PReLU(),
            UpsampleNet(in_channels, in_channels, upsample_factor),
            nn.Conv1d(in_channels, out_channels, kernel_size=15, dilation=1, padding=7),
        )

        self.condition_batchnorm2 = ConditionalBatchNorm1d(out_channels, condition_length)

        self.second_stack = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=15, dilation=1, padding=7),
        )

        self.residual1 = nn.Sequential(
            UpsampleNet(in_channels, in_channels, upsample_factor),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0),
        )
        # Block B
        self.condition_batchnorm3 = ConditionalBatchNorm1d(out_channels, condition_length)

        self.third_stack = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=15, dilation=4, padding=28),
        )

        self.condition_batchnorm4 = ConditionalBatchNorm1d(out_channels, condition_length)

        self.fourth_stack = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=15, dilation=8, padding=56),
        )

    def forward(self, enc_out, condition):
        inputs = enc_out

        outputs = self.condition_batchnorm1(inputs, condition)
        outputs = self.first_stack(outputs)
        outputs = self.condition_batchnorm2(outputs, condition)
        outputs = self.second_stack(outputs)

        residual_outputs = self.residual1(inputs) + outputs

        outputs = self.condition_batchnorm3(residual_outputs, condition)
        outputs = self.third_stack(outputs)
        outputs = self.condition_batchnorm4(outputs, condition)
        outputs = self.fourth_stack(outputs)

        outputs = outputs + residual_outputs

        return outputs


class Decoder(nn.Module):
    def __init__(self, num_filters, cond_length):
        super(Decoder, self).__init__()

        self.preprocess = nn.Conv1d(1, 512, kernel_size=15, padding=7)
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(512, 512, 1, cond_length),
                DecoderBlock(512, 512, 1, cond_length),
                DecoderBlock(512, 256, 2, cond_length),
                DecoderBlock(256, 256, 2, cond_length),
                DecoderBlock(256, 256, 2, cond_length),
                DecoderBlock(256, 128, 3, cond_length),
                DecoderBlock(128, 64, 5, cond_length),
            ]
        )

        self.postprocess = nn.Sequential(nn.Conv1d(64, num_filters + 1, kernel_size=15, padding=7))

        self.sigmoid = nn.Sigmoid()

    def forward(self, v, condition):
        inputs = self.preprocess(v)
        outputs = inputs
        for i, layer in enumerate(self.blocks):
            outputs = layer(outputs, condition)
        outputs = self.postprocess(outputs)

        direct_early = outputs[:, 0:1]
        late = outputs[:, 1:]

        late = self.sigmoid(late)

        return direct_early, late


class FINS(pl.LightningModule):
    def __init__(self, config):
        super(FINS, self).__init__()

        self.config = config

        self.rir_length = int(config.fins.rir_duration * config.sample_rate)
        self.min_snr, self.max_snr = config.fins.min_snr, config.fins.max_snr

        # Learned decoder input
        # ensure user set decoder_input_length in config correctly
        if self.rir_length % 120 == 0:
            # 120 is the net upsampling factor of the decoder. If the RIR length is divisible by 120, the decoder input length (set in config) should strictly be 1/120 of the RIR length.
            assert config.fins.decoder_input_length == self.rir_length // 120 
        else:
            # otherwise, the decoder input length (setin config) should be ceil(1/120 of the RIR length). The few additional samples this yields will be truncated in prediction.
            assert config.fins.decoder_input_length == int(np.ceil(self.rir_length / 120))
        self.decoder_input = nn.Parameter(torch.randn((1, 1, config.fins.decoder_input_length)))  # 1,1,400
        self.encoder = Encoder()

        self.decoder = Decoder(config.fins.num_filters, config.fins.noise_condition_length + config.fins.z_size)

        # Learned "octave-band" like filter
        self.filter = nn.Conv1d(
            config.fins.num_filters,
            config.fins.num_filters,
            kernel_size=config.fins.filter_order,
            stride=1,
            padding='same',
            groups=config.fins.num_filters,
            bias=False,
        )

        # Octave band pass initialization
        octave_filters = get_octave_filters()
        self.filter.weight.data = torch.FloatTensor(octave_filters)

        # self.filter.bias.data.zero_()

        # Mask for direct and early part
        mask = torch.zeros((1, 1, self.rir_length))
        mask[:, :, : self.config.fins.early_length] = 1.0
        self.register_buffer("mask", mask)
        self.output_conv = nn.Conv1d(config.fins.num_filters + 1, 1, kernel_size=1, stride=1)

        # Loss
        fft_sizes = [64, 512, 2048, 8192]
        hop_sizes = [32, 256, 1024, 4096]
        win_lengths = [64, 512, 2048, 8192]
        sc_weight = 1.0
        mag_weight = 1.0
        self.stft_loss_fn = MultiResolutionSTFTLoss(
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_lengths=win_lengths,
            sc_weight=sc_weight,
            mag_weight=mag_weight,
        ).to(self.device)

        self.recon_stft_loss_fn = MultiResolutionSTFTLoss(
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_lengths=win_lengths,
            sc_weight=sc_weight,
            mag_weight=mag_weight,
        ).to(self.device)



    def predict(self, x, stochastic_noise, noise_condition):
        """
        args:
            x : Reverberant speech. shape=(batch_size, 1, input_samples)
            stochastic_noise : Random normal noise for late reverb synthesis. shape=(batch_size, n_freq_bands, length_of_rir)
            noise_condition : Noise used for conditioning. shape=(batch_size, noise_cond_length)
        return
            rir: shape=(batch_size, 1, rir_samples)
        """
        b, _, _ = x.size()

        # Filter random noise signal
        filtered_noise = self.filter(stochastic_noise)

        # Encode the reverberated speech
        z = self.encoder(x)

        # Make condition vector
        condition = torch.cat([z, noise_condition], dim=-1)
   
        # Learnable decoder input. Repeat it in the batch dimension.
        decoder_input = self.decoder_input.repeat(b, 1, 1)

        # Generate RIR
        direct_early, late_mask = self.decoder(decoder_input, condition)

        # truncate the late mask and direct early to the length of the filtered noise (necessary if the RIR duration cannot be directly achieved by convolution, i.e., evenly divided by the decoder upsampling factors)
        late_mask = late_mask[:, :, : filtered_noise.size(-1)]
        direct_early = direct_early[:, :, : filtered_noise.size(-1)]

        # Apply mask to the filtered noise to get the late part
        late_part = filtered_noise * late_mask
     
        # Zero out sample beyond 2400 for direct early part
        direct_early = torch.mul(direct_early, self.mask)
        # Concat direct,early with late and perform convolution
        rir = torch.cat((direct_early, late_part), 1)

        # Sum
        rir = self.output_conv(rir)

        return rir

 
    def training_step(self, batch, batch_idx):
        loss_type = "train"

        _, _, _, _, enc_reverb_speech_wav, _, rir, stochastic_noise, noise_condition, _, _ = batch
 
        # convert speech wavs and noise to floats
        enc_reverb_speech_wav = enc_reverb_speech_wav.float()
        stochastic_noise = stochastic_noise.float()
        noise_condition = noise_condition.float()

        predicted_rir = self.predict(enc_reverb_speech_wav, stochastic_noise, noise_condition)
        predicted_rir = predicted_rir.squeeze(1)

    
        # Compute loss
        stft_loss_dict = self.stft_loss_fn(predicted_rir, rir)
        stft_loss = stft_loss_dict["total"]
        sc_loss = stft_loss_dict["sc_loss"].item()
        mag_loss = stft_loss_dict["mag_loss"].item()
        self.log("stft_loss_" + loss_type, stft_loss, sync_dist = True )

        if batch_idx % self.config.plot_every_n_steps == 0:
            self.plot_rirs(rir, predicted_rir, batch_idx, loss_type=loss_type)
        
        return stft_loss


    def validation_step(self, batch, batch_idx):
        loss_type = "val"

        _, _, _, _, enc_reverb_speech_wav, _, rir, stochastic_noise, noise_condition, _, _ = batch

        # convert speech wavs and noise to floats
        enc_reverb_speech_wav = enc_reverb_speech_wav.float()
        stochastic_noise = stochastic_noise.float()
        noise_condition = noise_condition.float()

        predicted_rir = self.predict(enc_reverb_speech_wav, stochastic_noise, noise_condition)
        predicted_rir = predicted_rir.squeeze(1)

        # Compute loss
        stft_loss_dict = self.stft_loss_fn(predicted_rir, rir)
        stft_loss = stft_loss_dict["total"]
        sc_loss = stft_loss_dict["sc_loss"].item()
        mag_loss = stft_loss_dict["mag_loss"].item()
        self.log("stft_loss_" + loss_type, stft_loss, sync_dist = True )

        self.plot_rirs(rir, predicted_rir, batch_idx, loss_type=loss_type)
        
        return stft_loss


    def test_step(self, batch, batch_idx):
       pass
    
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
        optimizer = optim.Adam(self.parameters(), lr=self.config.fins.lr, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config.fins.lr_step_size,
            gamma=self.config.fins.lr_decay_factor,
        )
        return [optimizer], [scheduler]

    