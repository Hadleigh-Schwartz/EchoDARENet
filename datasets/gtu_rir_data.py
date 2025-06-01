"""
Dataset class for Gthe GTU RIR dataset
https://github.com/mehmetpekmezci/gtu-rir/tree/master
"""

from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import soundfile as sf
import glob
import os
import pickle

class GTUIRDataset(Dataset):
    """
    GTU RIR data is stored in a pickle of .dat files so we need to read it first
    """
    def __init__(self, config, type="train", split_train_val_test_p=[80,10,10], device='cuda'):
        self.config = config
        self.data_file_path = Path(os.path.expanduser(self.config['datasets_path']), "gtu.pickle.dat")
        self.type = type

        # .dat fields. Taken from https://github.com/mehmetpekmezci/gtu-rir/blob/master/02.data/data_reader/read_data.py
        self.rir_data_field_numbers={
            "timestamp":0,
            "speakerMotorIterationNo":1,
            "microphoneMotorIterationNo":2,
            "speakerMotorIterationDirection":3,
            "currentActiveSpeakerNo":4,
            "currentActiveSpeakerChannelNo":5,
            "physicalSpeakerNo":6,
            "microphoneStandInitialCoordinateX":7,
            "microphoneStandInitialCoordinateY":8,
            "microphoneStandInitialCoordinateZ":9,
            "speakerStandInitialCoordinateX":10,
            "speakerStandInitialCoordinateY":11,
            "speakerStandInitialCoordinateZ":12,
            "microphoneMotorPosition":13,
            "speakerMotorPosition":14,
            "temperatureAtMicrohponeStand":15,
            "humidityAtMicrohponeStand":16,
            "temperatureAtMSpeakerStand":17,
            "humidityAtSpeakerStand":18,
            "tempHumTimestamp":19,
            "speakerRelativeCoordinateX":20,
            "speakerRelativeCoordinateY":21,
            "speakerRelativeCoordinateZ":22,
            "microphoneStandAngle":23,
            "speakerStandAngle":24,
            "speakerAngleTheta":25,
            "speakerAnglePhi":26,
            "mic_RelativeCoordinateX":27,
            "mic_RelativeCoordinateY":28,
            "mic_RelativeCoordinateZ":29,
            "mic_DirectionX":30,
            "mic_DirectionY":31,
            "mic_DirectionZ":32,
            "mic_Theta":33,
            "mic_Phi":34,
            "essFilePath":35,
            "roomId":36,
            "configId":37,
            "micNo":38, 
            "roomWidth":39,
            "roomHeight":40,
            "roomDepth":41, 
            "rt60":42, 
            "rirData":43 
        } 

        # load the data from the pickle file
        self.rir_data=[]  
        if os.path.exists(self.data_file_path):
            with open(self.data_file_path, 'rb') as f:
                self.rir_data = pickle.load(f)
        

        # divide the files by roomId, micNo, and physicalSpeakerNo
        divided_files = {}
        for file_id, file in enumerate(self.rir_data):
            room_id = file[self.rir_data_field_numbers["roomId"]]
            mic_no = file[self.rir_data_field_numbers["micNo"]]
            speaker_no = file[self.rir_data_field_numbers["physicalSpeakerNo"]]
            if room_id not in divided_files:
                divided_files[room_id] = {}
            if mic_no not in divided_files[room_id]:
                divided_files[room_id][mic_no] = {}
            if speaker_no not in divided_files[room_id][mic_no]:
                divided_files[room_id][mic_no][speaker_no] = []
            divided_files[room_id][mic_no][speaker_no].append(file_id)

        # randomly choose one file from each roomId, micNo, and physicalSpeakerNo combination
        self.rir_file_ids = []
        for room_id, mic_dict in divided_files.items():
            for mic_no, speaker_dict in mic_dict.items():
                for speaker_no, file_ids in speaker_dict.items():
                    if len(file_ids) > 0:
                        # randomly choose one file from the list
                        chosen_file_id = np.random.choice(file_ids)
                        self.rir_file_ids.append(chosen_file_id)

        self.max_data_len = len(self.rir_file_ids)
        self.samplerate = 44100

        # divide chosen files into train, val, and test sets
        idx_rand = np.random.permutation(self.rir_file_ids) # shuffle the indices of the RIR files to work with
        if type == "train" and "GTU" not in config.val_rir_datasets and "GTU" not in config.test_rir_datasets:
            self.split_ids = idx_rand
        elif type == "val" and "GTU" not in config.train_rir_datasets and "GTU" not in config.test_rir_datasets:
            self.split_ids = idx_rand
        elif type == "test" and "GTU" not in config.train_rir_datasets and "GTU" not in config.val_rir_datasets:
            self.split_ids = idx_rand
        else:
            self.split_train_val_test_p = np.array(np.int16(split_train_val_test_p))
            self.split_train_val_test = np.int16(np.round( np.array(self.split_train_val_test_p)/100 * self.max_data_len ))
            self.split_edge = np.cumsum(np.concatenate(([0],self.split_train_val_test)), axis=0)
        
            self.split_ids = []
            if self.type == "train":
                self.split_ids = idx_rand[self.split_edge[0]:self.split_edge[1]]
            elif self.type == "val":
                self.split_ids= idx_rand[self.split_edge[1]:self.split_edge[2]]
            elif self.type == "test":
                self.split_ids = idx_rand[self.split_edge[2]:self.split_edge[3]]

        # get excluded RIR filenames and remove them from split_ids
        exclude_filenames = self.get_exclude_rirs()
        self.split_ids = [i for i in self.split_ids if i not in exclude_filenames]
        self.device = device

    def get_exclude_rirs(self):
        """
        Get list of ids of RIRs that are excluded from the dataset based on their early reverb time

        Early reverb time is taken to coincide with the max value in the RIR
        """
        min_early_reverb = self.config["min_early_reverb"]
        exclude_rirs = []
        # iterate through all RIRs 
        for idx in self.split_ids:
            audio_data = self.rir_data[idx][self.rir_data_field_numbers["rirData"]]
            max_rir_idx = np.argmax(audio_data)
            max_rir_time = max_rir_idx / self.samplerate
            if max_rir_time < min_early_reverb:
                exclude_rirs.append(idx)
        return exclude_rirs

    def __len__(self):
        return len(self.split_ids)

    def __getitem__(self, idx):
        file_id = self.split_ids[idx]
        audio_data = self.rir_data[file_id][self.rir_data_field_numbers["rirData"]]
        return audio_data, file_id

 
def GTURDataloader(config_path, type="train"):
    return DataLoader(GTUIRDataset(config_path, type=type))