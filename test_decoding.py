"""
Ensure that the differentiable decoding implementations work as expected,
i.e., return the same results as the traditional decoding implementations.
"""

import soundfile
import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np
from colorama import Fore, Style
from scipy.io import wavfile
from scipy.signal import fftconvolve, resample

import torch

from decoding import TimeDomainDecodingLoss, CepstralDomainDecodingLoss

# append echo encoding parent dir to path
import time
import sys
import os
curr_dir = os.getcwd()
echo_dir = curr_dir.split("EchoDARENet")[0] 
sys.path.append(echo_dir)
from traditional_echo_hiding import encode, decode
from pyroom_sim import run_sim

class DummyModel(torch.nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=2)
        self.conv2  = torch.nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=2)
        self.conv3  = torch.nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=2)
        self.conv4  = torch.nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=2)
    def forward(self, x):
        # convolve the input with a filter
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


def test_decoding_loss():

    delays = [i*2 for i in range(20, 75)]
    amplitude = 0.8
    win_size = 2048
    cutoff_freq = None
    kernel = "bp"
    if kernel == "ts":
        pn_size = 4000
        assert win_size > pn_size
        assert amplitude < 0.1 # weird things happen if amplitude is too high
        pn = np.random.choice([-1, 1], size = pn_size) # generate a random pseudo-noise sequence
    else:
        pn = None
    decoding = "cepstrum"


    audio, SR = soundfile.read("../../audio_samples/ffts_audio_min7s/dartagnan03part3_88_dumas_0053.flac")
    # convert to 0-1 range, downsample to 16 kHz, and trim to 32777 because that's what NN uses
    audio = audio / np.max(np.abs(audio))
    if SR != 16000:
        audio = resample(audio, int(len(audio) * 16000 / SR))
        SR = 16000
    audio = audio[:32777] 

    # initialize a TimeDomainDecodingLoss object
    decoding_loss = TimeDomainDecodingLoss(delays, win_size, decoding , cutoff_freq, SR, 1e10)

    num_wins = len(audio) // win_size
    audio = audio[:num_wins * win_size]
    symbols = np.random.randint(0, len(delays), size = num_wins)

    # get the decoding error without reverb
    encoded = encode(audio, symbols, amplitude, delays, win_size, SR, kernel, pn = pn)
    pred_symbols, pred_symbols_autocepstrum  = decode(encoded, delays, win_size, SR, pn = pn, gt = symbols, plot = False, cutoff_freq = cutoff_freq)
    num_errs = np.sum(np.array(pred_symbols) != np.array(symbols))
    num_errs_autocepstrum  = np.sum(np.array(pred_symbols_autocepstrum ) != np.array(symbols))
    print(f"Num err symbols og: {num_errs}/{len(pred_symbols)}. Num err symbols autocep {num_errs }/{len(pred_symbols_autocepstrum )}")
    if decoding == "autocepstrum":
        num_errs_no_reverb = num_errs_autocepstrum
    else:
        num_errs_no_reverb = num_errs

    # add some reverb
    simulated_signals, _ = run_sim(encoded, SR,  [2.5, 3.73, 1.76], f"tradtl_temp_pyroom_simulated_audio/")
    case_num = 20
    simulated_case = list(simulated_signals.keys())[case_num]
    room_tradtl_encoded = simulated_signals[simulated_case]

    # get the decoding error in the face of reverberation
    post_simulated_tradtl_decoded, post_simulated_tradtl_decoded_autocepstrum = decode(room_tradtl_encoded, delays, win_size, SR, cutoff_freq = cutoff_freq, plot = False, gt = symbols)
    post_simulated_tradtl_err = np.sum(np.array(post_simulated_tradtl_decoded[:len(symbols)]) != np.array(symbols))
    post_simulated_tradtl_err_autocepstrum = np.sum(np.array(post_simulated_tradtl_decoded_autocepstrum[:len(symbols)]) != np.array(symbols))
    print(f"Tradtl enc - Total errors cepstrum after simulation: {post_simulated_tradtl_err}/{len(symbols)}. Total errors autocepstrum after simulation: {post_simulated_tradtl_err_autocepstrum}/{len(symbols)}.")
    if decoding == "autocepstrum":
        num_errs_reverb = post_simulated_tradtl_err_autocepstrum 
    else:
        num_errs_reverb = post_simulated_tradtl_err

    room_tradl_encoded_tensor = torch.tensor(room_tradtl_encoded, dtype = torch.float32)
    symbols = torch.tensor(symbols, dtype = torch.int64)

    # add batch dimensions
    room_tradl_encoded_tensor = room_tradl_encoded_tensor.unsqueeze(0).unsqueeze(0)
    symbols = symbols.unsqueeze(0)
    num_errs_no_reverb_batch = torch.tensor(num_errs_no_reverb, dtype = torch.float32).unsqueeze(0)
    num_errs_reverb_batch = torch.tensor(num_errs_reverb, dtype = torch.float32).unsqueeze(0)

    # do a forward pass through the decoding loss
    symbol_err_rate, tot_symbol_errs, avg_err_reduction = decoding_loss(room_tradl_encoded_tensor, symbols, num_errs_no_reverb_batch, num_errs_reverb_batch)
    print(f"Symbol error rate: {symbol_err_rate}")
    print(f"Total symbol errors: {tot_symbol_errs}")
    print(f"Average error reduction loss (1 - avg err reduction loss): {avg_err_reduction}")

    # train the dummy model using torch_decode as a loss function
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1000):  # loop over the dataset multiple times
        optimizer.zero_grad()
        output = model(room_tradl_encoded_tensor)  # add batch and channel dimensions
        symbol_err_rate, tot_symbol_errs, avg_err_reduction = decoding_loss(output, symbols, num_errs_no_reverb_batch, num_errs_reverb_batch)
        loss = avg_err_reduction
        loss.backward()
        optimizer.step()

def examine_softargmax():
  
    # set up decoding loss so we can use softargmax
    delays = [i*2 for i in range(20, 75)]
    amplitude = 0.8
    win_size = 2048
    cutoff_freq = None
    kernel = "bp"
    if kernel == "ts":
        pn_size = 4000
        assert win_size > pn_size
        assert amplitude < 0.1 # weird things happen if amplitude is too high
        pn = np.random.choice([-1, 1], size = pn_size) # generate a random pseudo-noise sequence
    else:
        pn = None
    decoding = "cepstrum"

    # create some test lists
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 9.1]
    b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    c = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11]
    a = torch.tensor(a, dtype = torch.float32).unsqueeze(0)
    b = torch.tensor(b, dtype = torch.float32).unsqueeze(0)
    c = torch.tensor(c, dtype = torch.float32).unsqueeze(0)

    # pass them through the softargmax function
    for beta in [10**b for b in range(11)]:
        decoding_loss = TimeDomainDecodingLoss(delays, win_size, decoding , cutoff_freq, 16000, beta)

        a_out = decoding_loss.softargmax(a)
        b_out = decoding_loss.softargmax(b)
        c_out = decoding_loss.softargmax(c)

        print(f"------ Beta: {beta} ------")
        print(f"a_out: {a_out}")
        print(f"b_out: {b_out}")
        print(f"c_out: {c_out}")
        print("--------------------------")


def get_cepstrum_windows(audio, win_size):
        num_wins = len(audio) // win_size
       
        win_cepstra = []
        for i in range(num_wins):
            win = audio[i * win_size : (i + 1) * win_size]

            fft = np.fft.fft(win)
            sqr_log_fft = np.log(np.abs(fft) + 0.00001)
            cepstrum = np.fft.ifft(sqr_log_fft)
            # only need the first half because second half is just the conjugate of the first half
            cepstrum = cepstrum[:len(cepstrum) // 2]
            win_cepstra.append(np.stack((np.real(cepstrum), np.imag(cepstrum))))
        return win_cepstra    

def test_decoding_loss2():
    delays = [i*2 for i in range(20, 75)]
    amplitude = 0.8
    win_size = 2048
    cutoff_freq = None
    kernel = "bp"
    if kernel == "ts":
        pn_size = 4000
        assert win_size > pn_size
        assert amplitude < 0.1 # weird things happen if amplitude is too high
        pn = np.random.choice([-1, 1], size = pn_size) # generate a random pseudo-noise sequence
    else:
        pn = None
    decoding = "cepstrum"


    audio, SR = soundfile.read("../../audio_samples/ffts_audio_min7s/dartagnan03part3_88_dumas_0053.flac")
    # convert to 0-1 range, downsample to 16 kHz, and trim to 32777 because that's what NN uses
    audio = audio / np.max(np.abs(audio))
    if SR != 16000:
        audio = resample(audio, int(len(audio) * 16000 / SR))
        SR = 16000
    audio = audio[:32777] 

    # initialize a TimeDomainDecodingLoss object
    decoding_loss = TimeDomainDecodingLoss(delays, win_size, decoding , cutoff_freq, SR, 1e10)

    num_wins = len(audio) // win_size
    audio = audio[:num_wins * win_size]
    symbols = np.random.randint(0, len(delays), size = num_wins)

    # get the decoding error without reverb
    encoded = encode(audio, symbols, amplitude, delays, win_size, SR, kernel, pn = pn)
    pred_symbols, pred_symbols_autocepstrum  = decode(encoded, delays, win_size, SR, pn = pn, gt = symbols, plot = False, cutoff_freq = cutoff_freq)
    num_errs = np.sum(np.array(pred_symbols) != np.array(symbols))
    num_errs_autocepstrum  = np.sum(np.array(pred_symbols_autocepstrum ) != np.array(symbols))
    print(f"Num err symbols og: {num_errs}/{len(pred_symbols)}. Num err symbols autocep {num_errs }/{len(pred_symbols_autocepstrum )}")
    if decoding == "autocepstrum":
        num_errs_no_reverb = num_errs_autocepstrum
    else:
        num_errs_no_reverb = num_errs

    # add some reverb
    simulated_signals, _ = run_sim(encoded, SR,  [2.5, 3.73, 1.76], f"tradtl_temp_pyroom_simulated_audio/")
    case_num = 20
    simulated_case = list(simulated_signals.keys())[case_num]
    room_tradtl_encoded = simulated_signals[simulated_case]

    # get the decoding error in the face of reverberation
    post_simulated_tradtl_decoded, post_simulated_tradtl_decoded_autocepstrum = decode(room_tradtl_encoded, delays, win_size, SR, cutoff_freq = cutoff_freq, plot = False, gt = symbols)
    post_simulated_tradtl_err = np.sum(np.array(post_simulated_tradtl_decoded[:len(symbols)]) != np.array(symbols))
    post_simulated_tradtl_err_autocepstrum = np.sum(np.array(post_simulated_tradtl_decoded_autocepstrum[:len(symbols)]) != np.array(symbols))
    print(f"Tradtl enc - Total errors cepstrum after simulation: {post_simulated_tradtl_err}/{len(symbols)}. Total errors autocepstrum after simulation: {post_simulated_tradtl_err_autocepstrum}/{len(symbols)}.")
    if decoding == "autocepstrum":
        num_errs_reverb = post_simulated_tradtl_err_autocepstrum 
    else:
        num_errs_reverb = post_simulated_tradtl_err

    room_tradtl_encoded = room_tradtl_encoded[:32777] # trim to 32777 because that's what NN uses

    room_tradl_encoded_tensor = torch.tensor(room_tradtl_encoded, dtype = torch.float32)
    symbols = torch.tensor(symbols, dtype = torch.int64)

    # add batch dimensions
    room_tradl_encoded_tensor = room_tradl_encoded_tensor.unsqueeze(0).unsqueeze(0)
    symbols = symbols.unsqueeze(0)
    num_errs_no_reverb_batch = torch.tensor(num_errs_no_reverb, dtype = torch.float32).unsqueeze(0)
    num_errs_reverb_batch = torch.tensor(num_errs_reverb, dtype = torch.float32).unsqueeze(0)

    # duplicate each tensor along batch dimension
    test_batch_size = 8
    room_tradl_encoded_tensor = room_tradl_encoded_tensor.repeat(test_batch_size, 1, 1)
    symbols = symbols.repeat(test_batch_size, 1)
    num_errs_no_reverb_batch = num_errs_no_reverb_batch.repeat(test_batch_size)
    num_errs_reverb_batch = num_errs_reverb_batch.repeat(test_batch_size)
    print(room_tradl_encoded_tensor.shape, symbols.shape, num_errs_no_reverb_batch.shape, num_errs_reverb_batch.shape)

    start = time.time()
    sym_err_rate, avg_err_reduction_loss, no_reverb_sym_err_rate, reverb_sym_err_rate = decoding_loss(room_tradl_encoded_tensor, symbols, num_errs_no_reverb_batch, num_errs_reverb_batch)
    end = time.time()
    print(f"Time domain results: {sym_err_rate}, {avg_err_reduction_loss}, {no_reverb_sym_err_rate}, {reverb_sym_err_rate}")

    # convert room_tradtl_encoded to cepstra
    win_cepstra = get_cepstrum_windows(room_tradtl_encoded, win_size)
    win_cepstra = np.dstack(win_cepstra)    
    win_cepstra_tensor = torch.tensor(win_cepstra, dtype = torch.float32)
    # reshape to 1, 2, 1024, 16
    win_cepstra_tensor = win_cepstra_tensor.unsqueeze(0)
    # duplicate to batches
    win_cepstra_tensor = win_cepstra_tensor.repeat(test_batch_size, 1, 1, 1)

    # initialize CepstralDomainDecodingLoss
    cepstral_decoding_loss = CepstralDomainDecodingLoss(delays, win_size, decoding , cutoff_freq, SR, 1e10)
    start = time.time()
    sym_err_rate, avg_err_reduction_loss, no_reverb_sym_err_rate, reverb_sym_err_rate = cepstral_decoding_loss(win_cepstra_tensor, symbols, num_errs_no_reverb_batch, num_errs_reverb_batch)
    end = time.time()
    print(f"Cepstral domain results: {sym_err_rate}, {avg_err_reduction_loss}, {no_reverb_sym_err_rate}, {reverb_sym_err_rate}")

test_decoding_loss2()