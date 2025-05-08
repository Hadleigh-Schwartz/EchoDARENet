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
            "timestamp":0,"speakerMotorIterationNo":1,"microphoneMotorIterationNo":2,"speakerMotorIterationDirection":3,"currentActiveSpeakerNo":4,"currentActiveSpeakerChannelNo":5,
            "physicalSpeakerNo":6,"microphoneStandInitialCoordinateX":7,"microphoneStandInitialCoordinateY":8,"microphoneStandInitialCoordinateZ":9,"speakerStandInitialCoordinateX":10,
            "speakerStandInitialCoordinateY":11,"speakerStandInitialCoordinateZ":12,"microphoneMotorPosition":13,"speakerMotorPosition":14,"temperatureAtMicrohponeStand":15,
            "humidityAtMicrohponeStand":16,"temperatureAtMSpeakerStand":17,"humidityAtSpeakerStand":18,"tempHumTimestamp":19,"speakerRelativeCoordinateX":20,"speakerRelativeCoordinateY":21,
            "speakerRelativeCoordinateZ":22,"microphoneStandAngle":23,"speakerStandAngle":24,"speakerAngleTheta":25,"speakerAnglePhi":26,"mic_RelativeCoordinateX":27,"mic_RelativeCoordinateY":28,
            "mic_RelativeCoordinateZ":29,"mic_DirectionX":30,"mic_DirectionY":31,"mic_DirectionZ":32,"mic_Theta":33,"mic_Phi":34,"essFilePath":35,
            "roomId":36,"configId":37,"micNo":38, 
            "roomWidth":39,"roomHeight":40,"roomDepth":41, 
            "rt60":42, 
            "rirData":43 
        } 

        # load the data from the pickle file
        self.rir_data=[]  
        if os.path.exists(self.data_file_path):
            with open(self.data_file_path, 'rb') as f:
                self.rir_data = pickle.load(f)
        
        self.max_data_len = len(self.rir_data)

        self.samplerate = 44100

        self.split_train_val_test_p = np.array(np.int16(split_train_val_test_p))
        self.split_train_val_test = np.int16(np.round( np.array(self.split_train_val_test_p)/100 * self.max_data_len ))
        self.split_edge = np.cumsum(np.concatenate(([0],self.split_train_val_test)), axis=0)
        self.idx_rand = np.random.RandomState(seed=config['random_seed']).permutation(self.max_data_len)

        self.split_ids = []
        if self.type == "train":
            self.split_ids = self.idx_rand[self.split_edge[0]:self.split_edge[1]]
        elif self.type == "val":
            self.split_ids= self.idx_rand[self.split_edge[1]:self.split_edge[2]]
        elif self.type == "test":
            self.split_ids = self.idx_rand[self.split_edge[2]:self.split_edge[3]]

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