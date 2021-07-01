import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2CTCTokenizer
import numpy

class LanguageClassifier(nn.Module):
    def __init__(self, out_classes=3):
        '''instantiates a 3-language classifier for Swedish, English and Arabic'''
        super().__init__()
        config = Wav2Vec2Config()

        self.encoder = Wav2Vec2Model(config).from_pretrained('facebook/wav2vec2-large-xlsr-53')

        self.avg_pooling = nn.AvgPool1d(kernel_size=4)
        self.FC = nn.Linear(124, out_classes)
        self.out = nn.Softmax()

    def forward(self, X):
        context = self.encoder(X)
        pooled = self.avg_pooling(context)
        linear = self.FC(pooled)
        return self.out(linear)


