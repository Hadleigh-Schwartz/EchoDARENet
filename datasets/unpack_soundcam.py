"""
Load SoundCam RIRs from their orginal .npy format and save them in a more efficient format/organization
for torch dataloader processing.

SoundCam RIRs are initially stored in large .npy files containing arrays of shape (num_configs, num_mics, num_rir_samples).
These take a long time to load in datasets, so we split them into one .npy file per config, containing arrays of shape (num_mics, num_rir_samples).
We split along the configuration dimension because of two reasons:
1) the RIRs are primarily affected by mic. This means the dataloader should just choose one config of a room randomly, and then use all of the mics for that config.
Thus, saving one file per config is more intuitive for the end goal of making dataloading faster.
2) The number of configurations is much larger than the number of mics, and is thus responsible for slowing down .npy loading in the original format.
"""
import glob
import numpy as np
import os

soundcam_data_path = "../Datasets/SoundCam"
output_path = "../Datasets/SoundCamFlat"
if not os.path.exists(output_path):
    os.makedirs(output_path)
subfolders = glob.glob(f"{soundcam_data_path}/*")
for subfolder in subfolders:
    subfolder_name = os.path.basename(subfolder)
    human_folders = glob.glob(f"{subfolder}/*")
    for human_folder in human_folders:
        human_folder_name = os.path.basename(human_folder)
        print(f"Processing {human_folder}")
        # create a new subfolder in the output path
        output_subfolder = os.path.join(output_path, subfolder_name, human_folder_name)
        print(f"Creating {output_subfolder}")
        os.makedirs(output_subfolder, exist_ok=True)
        npy_path = f"{human_folder}/deconvolved.npy"
        # load the .npy file
        data = np.load(npy_path)
        # get the number of configurations
        num_configs = data.shape[0]
        # save each configuration as a separate .npy file
        for i in range(num_configs):
            # get the data for the configuration
            config_data = data[i]
            # save the data as a .npy file
            output_file = os.path.join(output_subfolder, f"config{i}_deconvolved.npy")
            print(f"Saving {output_file}")
            np.save(output_file, config_data)

