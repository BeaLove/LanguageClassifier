import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2CTCTokenizer
import numpy

class LanguageClassifier(nn.Module):
    def __init__(self, out_classes=3, pool_kernel=5):
        '''instantiates a 3-language classifier for Swedish, English and Arabic'''
        super().__init__()
        config = Wav2Vec2Config(return_dict=False)

        self.encoder = Wav2Vec2Model(config).from_pretrained('facebook/wav2vec2-large-xlsr-53')

        self.avg_pooling = nn.AvgPool2d(kernel_size=pool_kernel)
        self.FC = nn.Linear(5916, out_classes)
        self.out = nn.Softmax()

    def forward(self, X):
        wav2vecout = self.encoder(X)
        context = wav2vecout.last_hidden_state
        pooled = self.avg_pooling(context)
        vector = torch.flatten(pooled)
        linear = self.FC(vector)
        return self.out(linear)

