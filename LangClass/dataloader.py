import torch
from torch.utils.data import Dataset
#from transformers import Wav2Vec2Processor
import torchaudio
import os

##NOTE: we sadly need a separate class for each dataset because of how the language label is extracted from the data
class Commonvoice(Dataset):
    def __init__(self, dataset_dir, sample_len=4):
        self.data_dir = dataset_dir
        self.sample_len = sample_len
        self.lang_code = []
        self.dataset = []
        #self.processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-large-xlsr-53')
        for root, dirs, files in os.walk(self.data_dir, topdown=True):
            for file in files:
                self.dataset.append(os.path.join(root, file))
                lang = root.split("_")[-1]
                self.lang_code.append(lang)
        self.code_to_idx = {"en": 0, "ar": 1, "sv": 2}
        self.lang_idx = [self.code_to_idx[lang] for lang in self.lang_code]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        path = self.dataset[item]
        wav, samplerate = torchaudio.load(path)
        clip_len = samplerate*self.sample_len
        target = self.lang_idx[item]
        return wav[0], target
        #zero-pad short clips:
    '''
        if wav.shape[1] < clip_len:
            pad_size = clip_len - wav.shape[1]
            pad = wav[:,:pad_size]
            sample = torch.cat((wav, pad), dim=1)
            if sample.shape[1] < clip_len:
                pad_size = clip_len - sample.shape[1]
                pad = wav[:,:pad_size]
                sample = torch.cat((sample, pad), dim=1)
        elif wav.shape[1] >= clip_len:
            sample = wav[:,:clip_len]
        if sample.shape[1] != clip_len:
            print(path, "sample wrong", wav.shape[1])
        target = self.lang_idx[item]
        return sample[0], target
        #out = sample[0] '''



class Voxlingua(Dataset):
    def __init__(self, dataset_dir, sample_len=4):
        self.data_dir = dataset_dir
        self.sample_len = sample_len
        self.lang_code = []
        self.dataset = []
        #self.processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-large-xlsr-53')
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
            sample = torch.cat((wav, pad), dim=1)
            if sample.shape[1] < clip_len:
                pad_size = clip_len - sample.shape[1]
                pad = wav[:,:pad_size]
                sample = torch.cat((sample, pad), dim=1)
        elif wav.shape[1] >= clip_len:
            sample = wav[:,:clip_len]
        if sample.shape[1] != clip_len:
            print(path, "sample wrong", wav.shape[1])
        target = self.lang_idx[item]
        #out = sample[0]
        return sample[0], target

class PadSequence():
    def __call__(self, batch):
        unpacked, lengths_packed = torch.nn.utils.rnn.pad_packed_sequence(batch, batch_first=True)
        return unpacked

if __name__== '__main__':
    """debug code"""
    dataset =  Commonvoice(dataset_dir='data/train')
    dataset.__getitem__(0)

