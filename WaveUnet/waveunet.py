import torch
#import torch.nn as nn
from torch import optim, nn
import pytorch_lightning as pl
from torchaudio.functional import highpass_biquad

from WaveUnet.crop import centre_crop
from WaveUnet.resample import Resample1d
from WaveUnet.conv import ConvLayer

from decoding import TimeDomainDecodingLoss

import auraloss # for MR-STFT loss 
import matplotlib.pyplot as plt # for diagnostics only

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

        # self.norm_audio = config["norm_audio"]
        self.alpha = config["WaveUnet"]["alpha"]
        self.soft_beta = config["WaveUnet"]["soft_beta"]
        self.decoding_loss = TimeDomainDecodingLoss(config["Encoding"]["delays"], config["Encoding"]["win_size"], config["Encoding"]["decoding"], config["Encoding"]["cutoff_freq"], config["sample_rate"], 
                                          softargmax_beta=self.soft_beta)
        

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
        # x, y, z, symbols, num_errs_no_reverb, num_errs_reverb = batch # reverberant speech, clean speech, RIR # should be all time domain
        _, _, _, y, x, _, _, _, _, symbols,  _= batch
        num_errs_no_reverb = torch.tensor(0)
        num_errs_reverb = torch.tensor(0)
        print(x.shape, y.shape, symbols.shape)

        x = x.float()
        y = y.float()

        # convert from (batch_size, num_samples) to (batch_size, 1, num_samples)
        # x = x[:, None, :].float()
        # y = y[:, None, :].float()
        # z = z[:, None, :].float()

        out  = self.forward(x)
        # if self.norm_audio:
        #     gen_speech  = out["speech"]
        #     gen_speech = gen_speech - torch.mean(gen_speech)
        #     gen_speech = gen_speech / torch.max(torch.abs(gen_speech))
        # else:
        #     gen_speech  = out["speech"]
        speechMSEloss = nn.functional.mse_loss(out["speech"], centre_crop(y, out["speech"]))
        symbol_err_rate, avg_err_reduction, gt_symbol_err_rate_no_reverb, gt_symbol_err_rate_reverb = self.decoding_loss(out["speech"], symbols, num_errs_no_reverb, num_errs_reverb)
 
        loss = symbol_err_rate * self.alpha +  speechMSEloss * (1 - self.alpha)
           
        # if batch_idx % 100 == 0:
        #     self.plot(x, y, z, out, "Train")
            
        self.log("train_loss", loss )
        self.log("train_avg_err_reduction", avg_err_reduction)
        self.log("train_symbol_error_rate", symbol_err_rate)
        self.log("train_speechMSEloss", speechMSEloss)
        self.log("train_gt_symbol_error_rate_no_reverb", gt_symbol_err_rate_no_reverb)
        self.log("train_gt_symbol_error_rate_reverb", gt_symbol_err_rate_reverb)
        return loss

    def validation_step(self, batch, batch_idx):
        # validation_step defines the train loop.
        # it is independent of forward (but uses it)
        # x, y, z, symbols, num_errs_no_reverb, num_errs_reverb  = batch # reverberant speech, clean speech, RIR # should be all time domain
        _, _, _, y, x, _, _, _, _, symbols,  _= batch
        print(x.shape, y.shape, symbols.shape)
        num_errs_no_reverb = torch.tensor(0)
        num_errs_reverb = torch.tensor(0)

        x = x.float()
        y = y.float()

        # convert from (batch_size, num_samples) to (batch_size, 1, num_samples)
        # x = x[:, None, :].float()
        # y = y[:, None, :].float()
        # z = z[:, None, :].float()

        out  = self.forward(x)
        speechMSEloss = nn.functional.mse_loss(out["speech"], centre_crop(y, out["speech"]))
        print(out["speech"].shape)
        symbol_err_rate, avg_err_reduction, gt_symbol_err_rate_no_reverb, gt_symbol_err_rate_reverb = self.decoding_loss(out["speech"], symbols, num_errs_no_reverb, num_errs_reverb)
        
        loss = symbol_err_rate * self.alpha +  speechMSEloss * (1 - self.alpha)

        # self.plot(x, y, z, out, "Val")
           
        self.log("val_loss", loss )
        self.log("val_avg_err_reduction", avg_err_reduction)
        self.log("val_symbol_error_rate", symbol_err_rate)
        self.log("val_speechMSEloss", speechMSEloss)
        self.log("val_gt_symbol_error_rate_no_reverb", gt_symbol_err_rate_no_reverb)
        self.log("val_gt_symbol_error_rate_reverb", gt_symbol_err_rate_reverb)

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
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, tight_layout=True)
        ax1.plot(x[0,0,:].cpu().squeeze().detach().numpy())
        ax2.plot(y[0,0,:].cpu().squeeze().detach().numpy())
        ax3.plot(out["speech"][0,0,:].cpu().squeeze().detach().numpy())
        ax1.title.set_text("Cropped Reverb Speech")
        ax2.title.set_text("Cropped Clean Speech")
        ax3.title.set_text("Predicted Clean Speech")
        tb = self.logger.experiment
        tb.add_figure(log_title, fig, global_step=self.global_step)
        plt.close()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer