import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2CTCTokenizer
import numpy

class LanguageClassifier(nn.Module):
    def __init__(self, out_classes=3, pool_kernel=5):
        '''instantiates a 3-language classifier for Swedish, English and Arabic'''
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-large-xlsr-53')
        self.freeze_pretrained(self.encoder)
        self.avg_pooling = nn.AvgPool2d(kernel_size=pool_kernel)
        self.fc = nn.Linear(9996, out_classes)


    def forward(self, X):

        wav2vecout = self.encoder(X)
        context = wav2vecout.last_hidden_state
        pooled = self.avg_pooling(context)
        vector = pooled.reshape(1,-1)
        linear = self.fc(vector)
        return linear

    def freeze_pretrained(self, layer):
        for param in layer.parameters():
            param.requires_grad = False