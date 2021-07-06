import torch
from torch.utils.data import Dataset
import torchaudio
import os

class SentenceData(Dataset):
    def __init__(self, dataset_dir, sample_len=5):
        self.data_dir = dataset_dir
        self.sample_len = sample_len
        self.lang_code = []
        self.dataset = []
        for root, dirs, files in os.walk(self.data_dir, topdown=True):
            for file in files:
                self.dataset.append(os.path.join(root, file))
                lang = root.split("_")[-1]
                self.lang_code.append(lang)
        self.code_to_idx = {"en": 0, "ar": 1, "sv": 2}
        self.lang_idx = [self.code_to_idx[lang] for lang in self.lang_code]
        ##test: delete after debug

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        path = self.dataset[item]
        wav, samplerate = torchaudio.load(path)
        clip_len = samplerate*self.sample_len
        #zero-pad short clips:
        if wav.shape[1] < clip_len:
            pad_size = clip_len - wav.shape[1]
            pad = wav[:,:pad_size]
            wav = torch.cat((wav, pad), axis=1)
        elif wav.shape[1] > clip_len:
            wav = wav[:,:clip_len]
        assert wav.shape[1] == 80000
        target = self.lang_idx[item]
        return wav, target

if __name__== '__main__':
    """debug code"""
    dataset =  SentenceData(dataset_dir='data/train')
    dataset.__getitem__(0)

