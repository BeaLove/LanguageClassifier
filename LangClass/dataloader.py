from torch.utils.data import Dataset
import torchaudio
import os

class SentenceData(Dataset):
    def __init__(self, dataset_dir):
        self.data_dir = dataset_dir
        dataset_arabic =[] #TODO get dataset
        dataset_eng = []
        dataset_swe = []
        self.dataset = dataset_arabic + dataset_eng + dataset_swe
        self.lang_idx = [1]*len(dataset_arabic) + [2]*len(dataset_eng) + [3]*len(dataset_swe)
        ##test: delete after debug
        self.dataset.append('SA1.WAV.wav')
        self.lang_idx.append(2)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        path = os.path.join(self.data_dir, self.dataset[item])
        wav = torchaudio.load(path)
        sample_rate = wav[1]
        target = self.lang_idx[item]
        sample = wav[0]
        #TODO limit sample length to known sentence length
        return sample, target

if __name__== '__main__':
    """debug code"""
    dataset =  SentenceData(dataset_dir='data')
    dataset.__getitem__(0)

