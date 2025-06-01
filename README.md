# Encoding-Aware Dereverberation 

## Setup
TODO
- [ ]  Export conda env and put here

## Datasets
The speech and RIR datasets used to train/validate the model is selected in the config file. The following datasets are supported:
- Speech
  - LibriSpeech - 16kHz speech data
  - HiFi Multispeaker - 48kHz speech data
  Note: Data can be either encoded on-the-fly in the dataloader or preencoded and stored locally. See Preencoding below.
- RIR:
  - MIT IR Survey - 32 kHz RIRs from real environments
  - HOMULA RIR - 48kHz RIRs from real environments 
  - SoundCam
  - EARS
  - Simulated IR - Generated at any sampling rate using generate_ir.py
  

### Downloads/Generation
- The [LibriSpeech](https://www.openslr.org/12) dataset and the [MIT RIR Survey](https://mcdermottlab.mit.edu/Reverb/IR_Survey.html) dataset are  automatically downloaded and be prepared the first time the models are trained, validated or tested on these datasets.

- Download the [Hi-Fi Multispeaker](https://arxiv.org/abs/2104.01497) dataset [here](https://www.openslr.org/109/). Unzip and place in the Datasets directory. The expected file structure is:
  ```
    Datasets/
    ├── hi_fi_tts_v0/
    │   ├── audio
    │   │  ├── <split_num>_<clean/other>/
    │   │  │   ├── <subsplit_num>
    │   │  │   │   ├── <filename>.flac
    │   │  │   │   ├── ...
    │   │  │   ├── ...
    │   │  ├── ....
  ```

- Download the Homula IR dataset [here](https://zenodo.org/records/10479726). Unzip, reorganize into a flat folder structure and place in the Datasets directory. The expected file structure is:
```
    Datasets/
    ├── homula_ir/
    │   ├── <hom_ir_name>.wav
    │   ├── ....
  ```

- Download the preprocessed subsets of the SoundCam dataset [here](https://purl.stanford.edu/xq364hd5023). Unzip and place each subset of data (each of which is a subfolder upon unzipping) into one parent directory and then run the dataset/unpack_soundcam.py script to re-organize them into .wav files in a flat directory structure.

- Download the EARS RIRs using the official download script [here](https://github.com/sp-uhh/ears_benchmark/blob/main/download_ears_reverb.sh). Comment out the lines in this script corresponding to download of the speech dataset if you only want the RIRs.

- Generate random simulated IRs using generate_ir.py and place the resulting folder in Datasets. The expected file structure is:
   ```
    Datasets/
    ├── simulated_irs/
    │   ├── <generated_ir_name>.wav
    │   ├── ....
  ```

### Pre-encoding Speech
Encoding the speech data on the fly in the dataloader can be time consuming and slow training. We thus provide an option to pre-encode the speech data, i.e., encode it with random symbols and save the encoded audio file to disk along with the selected symbols. To do so and use preencoded data for training, simply do the following: 
1. Update the config file to specify the underlying raw speech dataset to use (i.e., LibriSpeech or Hi-Fi), any encoding parameters in the Encoding section, and the path on disk to store encoded speech files.
2. Run preencode_speech.py, specifying the path to the config file and, optionally, a maximum number of encoded speech files to generate per split. Encoded data will be generated and saved according to the config file.
3. Make sure not change any encoding parameters or encoding data path in the config in between generation and training/testing. That said, the dataloader will warn you if this occurs.