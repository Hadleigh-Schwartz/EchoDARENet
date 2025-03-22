import torch
#import torch.nn as nn
from torch import optim, nn
import pytorch_lightning as pl
from torchaudio.functional import highpass_biquad

from WaveUnet.crop import centre_crop
from WaveUnet.resample import Resample1d
from WaveUnet.conv import ConvLayer

import auraloss # for MR-STFT loss 
import matplotlib.pyplot as plt # for diagnostics only

import numpy as np
import sys
import os

curr_dir = os.getcwd()
echo_dir = curr_dir.split("EchoDARENet")[0] 
sys.path.append(echo_dir)


class DecodingLoss(nn.Module):
    def __init__(self, delays, win_size, decoding, cutoff_freq, sample_rate, softargmax_beta=1e10):
        super(DecodingLoss, self).__init__()
        self.delays = delays
        self.win_size = win_size
        self.decoding = decoding
        self.cutoff_freq = cutoff_freq
        self.sample_rate = sample_rate
        self.softargmax_beta = softargmax_beta

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
        tot_symbol_errs = 0 # total symbol errors (differentiable)
        total_err_reduction = 0 # accrue total error reduction (differentiable)
        num_symbols_per_audio = symbols_batch.shape[1]
        max_audio_len = num_symbols_per_audio * self.win_size
        for audio_idx in range(audio_batch.shape[0]):
            audio = audio_batch[audio_idx, 0, :max_audio_len]
            symbols = symbols_batch[audio_idx]
            num_errs_no_reverb = num_errs_no_reverb_batch[audio_idx]
            num_errs_reverb = num_errs_reverb_batch[audio_idx]
            num_wins = audio.shape[0] // self.win_size
            curr_audio_num_err_symbols = 0
            pred_symbols = []
            fig, axes = plt.subplots(2, 1, figsize=(10, 6), tight_layout=True)
            axes[0].plot(audio.cpu().numpy())
            if self.cutoff_freq is not None: # high-pass filter the window 
                audio = highpass_biquad(audio,  self.cutoff_freq , self.sample_rate) # FILTERING IS WRONG???
            axes[1].plot(audio.cpu().numpy())
            axes[1].set_title(f"Filtered Audio - {self.cutoff_freq} Hz cutoff")
            plt.savefig(f"audio_{audio_idx}.png")
            plt.close(fig)
            
            for i in range(num_wins):
                win = audio[i * self.win_size: (i + 1) * self.win_size]

                if self.decoding == "cepstrum":
                    cepstrum = self.torch_get_cepstrum(win)
                    cep_vals = cepstrum[self.delays]
                    max_val = self.softargmax(cep_vals)
                    cep_loss_val = torch.clamp(torch.abs(max_val - symbols[i]), min = 0, max = 1)
                    curr_audio_num_err_symbols= curr_audio_num_err_symbols + cep_loss_val
                    pred_symbols.append(int(max_val.item()))

                    # # get actual decoded symbol (non-differentiable, for debug)
                    # cep_vals_nondiff = cep_vals.clone()
                    # cep_vals_nondiff = cep_vals_nondiff.detach().cpu().numpy()
                    # max_val_nondiff = np.argmax(cep_vals_nondiff)
                    # if max_val_nondiff != symbols[i]:
                    #     tot_symbol_errors += 1 
                else:
                    autocepstrum = self.torch_get_autocepstrum(win)
                    autocepstrum_vals  = autocepstrum[self.delays]
                    max_autocepstrum_val  = self.softargmax(autocepstrum_vals )
                    autocep_loss_val = torch.clamp(torch.abs(max_autocepstrum_val - symbols[i]), min = 0, max = 1)
                    curr_audio_num_err_symbols = curr_audio_num_err_symbols + autocep_loss_val
                    pred_symbols.append(int(max_autocepstrum_val.item()))
                    
                    # autocep_vals_nondiff = autocepstrum_vals.clone()
                    # autocep_vals_nondiff = autocep_vals_nondiff.detach().cpu().numpy()
                    # max_autocepstrum_val_nondiff = np.argmax(autocep_vals_nondiff)
                    # if max_autocepstrum_val_nondiff != symbols[i]:
                    #     tot_symbol_errors += 1 
                    
            print(pred_symbols) 
            print(symbols.detach().cpu().numpy().tolist())
            print('-----------------')
            tot_symbol_errs = tot_symbol_errs + curr_audio_num_err_symbols
            curr_audio_err_reduction = ((num_errs_reverb - num_errs_no_reverb) - (curr_audio_num_err_symbols - num_errs_no_reverb)) / (num_errs_reverb - num_errs_no_reverb)
            total_err_reduction = total_err_reduction + (1 - curr_audio_err_reduction)

        symbol_err_rate = tot_symbol_errs / (audio_batch.shape[0] * num_symbols_per_audio)
        avg_err_reduction = total_err_reduction / audio_batch.shape[0]
        return symbol_err_rate, tot_symbol_errs, avg_err_reduction

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


class UpsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride, depth, conv_type, res):
        super(UpsamplingBlock, self).__init__()
        assert(stride > 1)

        # CONV 1 for UPSAMPLING
        if res == "fixed":
            self.upconv = Resample1d(n_inputs, 15, stride, transpose=True)
        else:
            self.upconv = ConvLayer(n_inputs, n_inputs, kernel_size, stride, conv_type, transpose=True)

        self.pre_shortcut_convs = nn.ModuleList([ConvLayer(n_inputs, n_outputs, kernel_size, 1, conv_type)] +
                                                [ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type) for _ in range(depth - 1)])

        # CONVS to combine high- with low-level information (from shortcut)
        
        # original, using shortcuts
        self.post_shortcut_convs = nn.ModuleList([ConvLayer(n_outputs + n_shortcut, n_outputs, kernel_size, 1, conv_type)] +
                                                 [ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type) for _ in range(depth - 1)])

        # new, avoiding shortcuts (doesn't work)
        #self.post_shortcut_convs = nn.ModuleList([ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type)] +
        #                                         [ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type) for _ in range(depth - 1)])



    def forward(self, x, shortcut):
        # UPSAMPLE HIGH-LEVEL FEATURES
        upsampled = self.upconv(x)

        for conv in self.pre_shortcut_convs:
            upsampled = conv(upsampled)

        # Prepare shortcut connection
        combined = centre_crop(shortcut, upsampled) # original, using shortcuts
        #combined = upsampled # new, avoiding shortcuts (doesn't work)

        # Combine high- and low-level features
        for conv in self.post_shortcut_convs:
            combined = conv(torch.cat([combined, centre_crop(upsampled, combined)], dim=1)) # original, using shortcuts
            #combined = conv(upsampled) # new, avoiding shortcuts (doesn't work)
        return combined

    def get_output_size(self, input_size):
        curr_size = self.upconv.get_output_size(input_size)

        # Upsampling convs
        for conv in self.pre_shortcut_convs:
            curr_size = conv.get_output_size(curr_size)

        # Combine convolutions
        for conv in self.post_shortcut_convs:
            curr_size = conv.get_output_size(curr_size)

        return curr_size

class DownsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride, depth, conv_type, res):
        super(DownsamplingBlock, self).__init__()
        assert(stride > 1)

        self.kernel_size = kernel_size
        self.stride = stride

        # CONV 1
        self.pre_shortcut_convs = nn.ModuleList([ConvLayer(n_inputs, n_shortcut, kernel_size, 1, conv_type)] +
                                                [ConvLayer(n_shortcut, n_shortcut, kernel_size, 1, conv_type) for _ in range(depth - 1)])

        self.post_shortcut_convs = nn.ModuleList([ConvLayer(n_shortcut, n_outputs, kernel_size, 1, conv_type)] +
                                                 [ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type) for _ in
                                                  range(depth - 1)])

        # CONV 2 with decimation
        if res == "fixed":
            self.downconv = Resample1d(n_outputs, 15, stride) # Resampling with fixed-size sinc lowpass filter
        else:
            self.downconv = ConvLayer(n_outputs, n_outputs, kernel_size, stride, conv_type)

    def forward(self, x):
        # PREPARING SHORTCUT FEATURES
        shortcut = x
        for conv in self.pre_shortcut_convs:
            shortcut = conv(shortcut)

        # PREPARING FOR DOWNSAMPLING
        out = shortcut
        for conv in self.post_shortcut_convs:
            out = conv(out)

        # DOWNSAMPLING
        out = self.downconv(out)

        return out, shortcut

    def get_input_size(self, output_size):
        curr_size = self.downconv.get_input_size(output_size)

        for conv in reversed(self.post_shortcut_convs):
            curr_size = conv.get_input_size(curr_size)

        for conv in reversed(self.pre_shortcut_convs):
            curr_size = conv.get_input_size(curr_size)
        return curr_size

class Waveunet(pl.LightningModule):
    def __init__(self, num_inputs, num_channels, num_outputs, instruments, kernel_size_down, kernel_size_up, target_output_size, conv_type, res, separate=False, depth=1, strides=2, learning_rate=0.0001, 
    config = None):
        super(Waveunet, self).__init__()
        # TODO: Initialize values based on config
        # TODO: make option of no decoding

        if config is not None:
            if config["echo_encode"] == False:
                raise Exception("Currently cannot use WaveUnet if config is not using echo encoding")
            self.decoding_loss = DecodingLoss(config["Encoding"]["delays"], config["Encoding"]["win_size"], config["Encoding"]["decoding"], config["Encoding"]["cutoff_freq"], config["sample_rate"])
        else:
            raise Exception("Need config to decode")


        self.name = "Waveunet"
        self.num_levels = len(num_channels)
        self.strides = strides
        self.kernel_size_down = kernel_size_down
        self.kernel_size_up = kernel_size_up
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.depth = depth
        self.instruments = instruments
        self.separate = separate
        self.learning_rate = learning_rate
        print('learning_rate = ' + str(learning_rate))

        #MR-STFT loss
        fft_sizes       = [16, 128, 512, 2048]
        hop_sizes       = [ 8,  64, 256, 1024]
        win_lengths     = [16, 128, 512, 2048]
        self.mrstftLoss = auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=fft_sizes, hop_sizes=hop_sizes, win_lengths=win_lengths)

        # Only odd filter kernels allowed
        assert(kernel_size_down % 2 == 1)
        assert(kernel_size_up   % 2 == 1)

        self.waveunets = nn.ModuleDict()

        model_list = instruments if separate else ["ALL"]
        # Create a model for each source if we separate sources separately, otherwise only one (model_list=["ALL"])
        for instrument in model_list:
            module = nn.Module()

            module.downsampling_blocks = nn.ModuleList()
            module.upsampling_blocks = nn.ModuleList()

            for i in range(self.num_levels - 1):
                in_ch = num_inputs if i == 0 else num_channels[i]

                module.downsampling_blocks.append(
                    DownsamplingBlock(in_ch, num_channels[i], num_channels[i+1], kernel_size_down, strides, depth, conv_type, res))

            for i in range(0, self.num_levels - 1):
                module.upsampling_blocks.append(
                    UpsamplingBlock(num_channels[-1-i], num_channels[-2-i], num_channels[-2-i], kernel_size_up, strides, depth, conv_type, res))

            module.bottlenecks = nn.ModuleList(
                [ConvLayer(num_channels[-1], num_channels[-1], kernel_size_down, 1, conv_type) for _ in range(depth)])

            # Output conv
            outputs = num_outputs if separate else num_outputs * len(instruments)
            module.output_conv = nn.Conv1d(num_channels[0], outputs, 1)

            self.waveunets[instrument] = module

        self.set_output_size(target_output_size)

    def set_output_size(self, target_output_size):
        self.target_output_size = target_output_size

        self.input_size, self.output_size = self.check_padding(target_output_size)
        print("Using valid convolutions with " + str(self.input_size) + " inputs and " + str(self.output_size) + " outputs")

        assert((self.input_size - self.output_size) % 2 == 0)
        self.shapes = {"output_start_frame" : (self.input_size - self.output_size) // 2,
                       "output_end_frame" : (self.input_size - self.output_size) // 2 + self.output_size,
                       "output_frames" : self.output_size,
                       "input_frames" : self.input_size}

    def check_padding(self, target_output_size):
        # Ensure number of outputs covers a whole number of cycles so
        # each output in the cycle is weighted equally during training
        bottleneck = 1

        while True:
            out = self.check_padding_for_bottleneck(bottleneck, target_output_size)
            if out is not False:
                return out
            bottleneck += 1

    def check_padding_for_bottleneck(self, bottleneck, target_output_size):
        module = self.waveunets[[k for k in self.waveunets.keys()][0]]
        try:
            curr_size = bottleneck
            for idx, block in enumerate(module.upsampling_blocks):
                curr_size = block.get_output_size(curr_size)
            output_size = curr_size

            # Bottleneck-Conv
            curr_size = bottleneck
            for block in reversed(module.bottlenecks):
                curr_size = block.get_input_size(curr_size)
            for idx, block in enumerate(reversed(module.downsampling_blocks)):
                curr_size = block.get_input_size(curr_size)

            assert(output_size >= target_output_size)
            return curr_size, output_size
        except AssertionError as e:
            return False

    def forward_module(self, x, module):
        '''
        A forward pass through a single Wave-U-Net (multiple Wave-U-Nets might be used, one for each source)
        :param x: Input mix
        :param module: Network module to be used for prediction
        :return: Source estimates
        '''
        shortcuts = []
        out = x

        # DOWNSAMPLING BLOCKS
        for block in module.downsampling_blocks:
            out, short = block(out)
            shortcuts.append(short)

        # BOTTLENECK CONVOLUTION
        for conv in module.bottlenecks:
            out = conv(out)

        # UPSAMPLING BLOCKS
        for idx, block in enumerate(module.upsampling_blocks):
            out = block(out, shortcuts[-1 - idx])

        # OUTPUT CONV
        out = module.output_conv(out)
        if not self.training:  # At test time clip predictions to valid amplitude range
            out = out.clamp(min=-1.0, max=1.0)
        return out

    def forward(self, x, inst=None):
        curr_input_size = x.shape[-1]
        #print("********************")
        #print("input shape = " + str(x.shape))
        #print("curr_input_size = " + str(curr_input_size))
        #print("self.input_size = " + str(self.input_size))
        #print("self.output_size = " + str(self.output_size))
        #print("********************")
        assert(curr_input_size == self.input_size) # User promises to feed the proper input himself, to get the pre-calculated (NOT the originally desired) output size

        if self.separate:
            return {inst : self.forward_module(x, self.waveunets[inst])}
        else:
            assert(len(self.waveunets) == 1)
            out = self.forward_module(x, self.waveunets["ALL"])

            out_dict = {}
            for idx, inst in enumerate(self.instruments):
                out_dict[inst] = out[:, idx * self.num_outputs:(idx + 1) * self.num_outputs]
            return out_dict
   
    def training_step(self, batch, batch_idx):
        x, y, z, symbols, num_errs_no_reverb, num_errs_reverb = batch # reverberant speech, clean speech, RIR # should be all time domain

        # convert from (batch_size, num_samples) to (batch_size, 1, num_samples)
        x = x[:, None, :].float()
        y = y[:, None, :].float()
        z = z[:, None, :].float()

        out  = self.forward(x)
    
        speechMSEloss = nn.functional.mse_loss(out["speech"], centre_crop(y, out["speech"]))
        err_rate, tot_symbol_errors = self.torch_decode(out["speech"], symbols, num_errs_no_reverb, num_errs_reverb)
        dec_loss =  err_rate
        dec_symbol_errors = tot_symbol_errors

        loss = dec_loss * 0.05 +  speechMSEloss * 0.95
           
        if batch_idx % 100 == 0:
            self.plot(x, y, z, out, "Train")
            
        self.log("train_loss", loss )
        self.log("train_decodeloss", dec_loss)
        self.log("train_decsymbolerrors", dec_symbol_errors)
        self.log("train_speechMSEloss", speechMSEloss)
        return loss

    def validation_step(self, batch, batch_idx):
        # validation_step defines the train loop.
        # it is independent of forward (but uses it)
        x, y, z, symbols, num_errs_no_reverb, num_errs_reverb  = batch # reverberant speech, clean speech, RIR # should be all time domain

        # convert from (batch_size, num_samples) to (batch_size, 1, num_samples)
        x = x[:, None, :].float()
        y = y[:, None, :].float()
        z = z[:, None, :].float()

        out  = self.forward(x)
        speechMSEloss = nn.functional.mse_loss(out["speech"], centre_crop(y, out["speech"]))
        err_rate,  tot_symbol_errors = self.torch_decode(out["speech"], symbols, num_errs_no_reverb, num_errs_reverb)
        dec_loss =  err_rate
        dec_symbol_errors = tot_symbol_errors
        loss = dec_loss * 0.05  +  speechMSEloss * 0.95

        self.plot(x, y, z, out, "Val")
           
        self.log("val_loss", loss)
        self.log("val_speechMSEloss", speechMSEloss )
        self.log("val_decodeloss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        # test_step for trainer.test()
        # it is independent of forward (but uses it)
        x, y, z,  symbols, baseline_num_errors = batch # reverberant speech, clean speech, RIR # should be all time domain

        # convert from (batch_size, num_samples) to (batch_size, 1, num_samples)
        x = x[:, None, :].float()
        y = y[:, None, :].float()
        z = z[:, None, :].float()

        out  = self.forward(x)
        speechMSEloss = nn.functional.mse_loss(out["speech"], centre_crop(y, out["speech"]))
        #rirMSEloss    = nn.functional.mse_loss(out["rir"], z)
        #rirMRSTFTloss = self.mrstftLoss(out["rir"], z)/400 # NOTE THE SCALE FACTOR!!  
        #loss = speechMSEloss + rirMSEloss
        #loss = speechMESloss + 5.0*rirMSEloss
        loss = speechMSEloss
        #loss = speechMSEloss + rirMRSTFTloss
        
        # Try nn.functional.l1_loss() for the RIR
        # Try a multiresolution FFT loss per Chrisitan's auraloss library

        # self.log("loss", {'test': loss })
        self.log("test_loss", loss )
        self.log("test_speechMSEloss", speechMSEloss )
        #self.log("test_rirMSEloss", rirMSEloss )
        #self.log("test_rirMRSTFTloss", rirMRSTFTloss )

        return loss

    def plot(self, x, y, z, out, log_title):
        fh = plt.figure()
        x = centre_crop(x, out["speech"])
        y = centre_crop(y, out["speech"])
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
        fig.set_size_inches(24, 4.8)
        ax1.plot(x[0,0,:].cpu().squeeze().detach().numpy())
        ax2.plot(y[0,0,:].cpu().squeeze().detach().numpy())
        ax3.plot(z[0,0,:].cpu().squeeze().detach().numpy())
        ax4.plot(out["speech"][0,0,:].cpu().squeeze().detach().numpy())
        ax5.plot(out["rir"][0,0,:].cpu().squeeze().detach().numpy())
        ax1.title.set_text("Cropped Reverb Speech")
        ax2.title.set_text("Cropped Clean Speech")
        ax3.title.set_text("GT RIR")
        ax4.title.set_text("Predicted Clean Speech")
        ax5.title.set_text("Predicted RIR")
        ax3.set_xlim(3100, 3300)
        tb = self.logger.experiment
        tb.add_figure(log_title, fig, global_step=self.global_step)
        plt.close()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer