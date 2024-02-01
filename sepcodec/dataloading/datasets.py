
import pandas as pd
from torch.utils.data import Dataset
import soundfile as sf
from encodec.utils import convert_audio
import numpy as np
import torch

import os


class SourceSeparationDataset(Dataset):

    def __init__(self, target_sample_rate=32000, target_length_s=6) -> None:
        super().__init__()
        self.target_sample_rate = target_sample_rate
        self.target_length_s = target_length_s
        
        self.annotations=[]

        self.target_length_n = int(
            self.target_length_s * self.target_sample_rate)

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        return {
            'mix': None,
            'target_stem': None,
            'conditioning': {
                'class_idx': None,
                'audio_query': None,
                'cluster_idx': None,
            }
        }

    def load_random_audio_chunk(self, path, start_idx=None, return_idx=False):
        '''
        Loads a random chunk of audio from a file. If start_idx is not specified, a random start index is chosen.'''

        target_samples = self.target_length_n
        target_sample_rate = self.target_sample_rate
        extension = path.split(".")[-1]
        try:
            info = sf.info(path)
            sample_rate = info.samplerate
        except:
            return None
        if extension == "mp3":
            n_frames = info.frames - 8192
        else:
            n_frames = info.frames

        new_target_samples = int(
            target_samples * sample_rate / target_sample_rate)

        if n_frames < new_target_samples:
            return None

        if start_idx is None:
            start_idx = np.random.randint(
                low=0, high=n_frames - new_target_samples)
        else:
            start_idx = start_idx

        waveform, sample_rate = sf.read(
            path, start=start_idx, stop=start_idx + new_target_samples, dtype='float32', always_2d=True)

        waveform = torch.Tensor(waveform.transpose())
        audio = convert_audio(
            waveform, sample_rate, target_sample_rate, 1)

        if return_idx:
            return audio, start_idx, start_idx + new_target_samples
        else:
            return audio

    def load_audio_and_split_in_chunks(self, path):
        target_samples = self.target_length_n
        target_sample_rate = self.target_sample_rate
        info = sf.info(path)
        sample_rate = info.samplerate

        waveform, sample_rate = sf.read(
            path, dtype='float32', always_2d=True)

        waveform = torch.Tensor(waveform.transpose())
        encodec_audio = convert_audio(
            waveform, sample_rate, target_sample_rate, 1)

        # ssplit audio into chunks of target_samples
        chunks = torch.split(encodec_audio, target_samples, dim=1)
        audio = torch.cat(chunks[:-1])  # drop the last one to avoid padding

        return audio

    def load_full_audio(self, path):
        target_sample_rate = self.target_sample_rate
        info = sf.info(path)
        sample_rate = info.samplerate

        waveform, sample_rate = sf.read(
            path, dtype='float32', always_2d=True)

        waveform = torch.Tensor(waveform.transpose())
        encodec_audio = convert_audio(
            waveform, sample_rate, target_sample_rate, 1)

        return encodec_audio
    

class Musdb18HQ(SourceSeparationDataset):
    
    def __init__(self, annotations = None, target_sample_rate=32000, target_length_s=6, train = True, transform = True) -> None:
        super().__init__(target_sample_rate=target_sample_rate, target_length_s=target_length_s)
        self.annotations = annotations
        self.classes = ["bass", "drums", "other", "vocals",'accompaniment']
        self.n_classes = len(self.classes)
        self.classes_to_idx = {c: i for i, c in enumerate(self.classes)}
        
    def sample_target_class(self):
        return np.random.choice(self.classes)
    
    def __getitem__(self, idx):
        folder_path = self.annotations.iloc[idx]['folder_path']
        # build a dict with the paths to the stems
        target_class = self.sample_target_class()
        target_class_idx = self.classes_to_idx[target_class]
        target_class_onehot = np.zeros(self.n_classes)
        target_class_onehot[target_class_idx] = 1
        
        target_audio, start_idx, end_idx = self.load_random_audio_chunk(os.path.join(folder_path, f'{target_class}.wav'), return_idx=True)
        
        # make a numpy array of shape (n_classes, target_length_n)
        mix_stems = np.zeros((self.n_classes, self.target_length_n))
        
        for i, stem in enumerate(self.classes):
            if stem == target_class:
                mix_stems[i] = target_audio
            else:
                mix_stems[i] = self.load_random_audio_chunk(os.path.join(folder_path, f'{stem}.wav'), start_idx=start_idx)
        
        # remixing augmentation here when clearer
        # for now:
        
        mix = np.sum(mix_stems, axis=0)
        
        mix = torch.Tensor(mix)
        target_audio = torch.Tensor(target_audio)
        target_class_idx = torch.Tensor([target_class_idx])
        
        return {
            'mix': mix,
            'target_stem': target_audio,
            'conditioning': {
                'class_idx': target_class_idx,
                'audio_query': torch.zeros(1),
                'cluster_idx': torch.zeros(1),
            }
        }
        
        