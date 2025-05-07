"""
Generate random simulated impulse responses (IRs) using PyRoomAcoustics
"""

import pyroomacoustics as pra
import numpy as np
import os
import random
from tqdm import tqdm
import soundfile as sf
import librosa

# Parameters
NUM_IRS = 1000
IR_DURATION = 1.0  # seconds
SAMPLE_RATE = 44100  # Hz
OUTPUT_DIR = "generated_irs"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_random_room():
    
    while True:
        # Random room dimensions (in meters)
        room_dim = np.random.uniform(low=3.0, high=50.0, size=(3,))

        # Random RT60 (reverberation time) between 0.2s and 1.2s
        rt60 = np.random.uniform(low=0.2, high=1.2)
        try:
            # Compute the absorption coefficient for the given RT60
            e_absorption, _ = pra.inverse_sabine(rt60, room_dim)
        except ValueError:  # avoid invalid RT60 values, which cause ValueError: evaluation of parameters failed. room may be too large for required RT60.
            continue
        break

    # Create room
    room = pra.ShoeBox(
        room_dim,
        fs=SAMPLE_RATE,
        materials=pra.Material(e_absorption),
        # max_order=17  # maximum number of reflections
    )
    
    # Random source position
    source_position = np.random.uniform(low=1.0, high=room_dim - 1.0)
    
    # Random microphone position
    mic_position = np.random.uniform(low=1.0, high=room_dim - 1.0)

    # Add source and microphone
    room.add_source(source_position)
    room.add_microphone_array(
        pra.MicrophoneArray(mic_position.reshape(3,1), room.fs)
    )

    return room

def generate_random_irs():
    # Generate IRs
    for i in tqdm(range(NUM_IRS), desc="Generating IRs"):
        room = generate_random_room()
        room.compute_rir()
        
        # Extract the first impulse response (source 0 to mic 0)
        ir = room.rir[0][0]
        
        # Ensure fixed length IR
        num_samples = int(IR_DURATION * SAMPLE_RATE)
        if len(ir) < num_samples:
            ir = np.pad(ir, (0, num_samples - len(ir)))
        else:
            ir = ir[:num_samples]
        
        # Save IR to wav file
        ir_filename = os.path.join(OUTPUT_DIR, f"ir_{i:04d}.wav")
        sf.write(ir_filename, ir, SAMPLE_RATE)

    print(f"Done! Generated {NUM_IRS} impulse responses in '{OUTPUT_DIR}'")

def test_irs():
    # load an audio file and convolve it with the generated IRs
    audio, audio_sr = sf.read("Datasets/hi_fi_tts_v0/audio/11614_other/10547/thousandnights8_11_anonymous_0069.flac")
    audio = librosa.resample(audio, orig_sr=audio_sr, target_sr=SAMPLE_RATE)
    ir_files = [os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR) if f.endswith('.wav')]
    random_ir_files = random.sample(ir_files, 10) # randomly select 10 IRs

    # convolve the audio with the IRs
    for ir_file in random_ir_files:
        ir, ir_sr = sf.read(ir_file)
        convolved_audio = np.convolve(audio, ir, mode='full')
        output_filename = f"convolved_{os.path.basename(ir_file)}.wav"
        sf.write(output_filename, convolved_audio, SAMPLE_RATE)
        print(f"Saved convolved audio to '{output_filename}'")

    # save original too for comparison
    sf.write(os.path.join(OUTPUT_DIR, "original_audio.wav"), audio, SAMPLE_RATE)
    print(f"Saved original audio to '{os.path.join(OUTPUT_DIR, 'original_audio.wav')}'")

if __name__ == "__main__":
    generate_random_irs()
    # test_irs() # optional, create and listen to the convolved audio files to verify the IRs