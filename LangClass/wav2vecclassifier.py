
import torch.nn as nn
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Model, Wav2Vec2Processor


class LanguageClassifier(nn.Module):
    def __init__(self, out_classes=3):
        '''instantiates a 3-language classifier for Swedish, English and Arabic'''
        super().__init__()
        #self.processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-large-xlsr-53')
        #self.encoder = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-large-xlsr-53')
        self.encoder = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-large-xlsr-53')
        self.freeze_pretrained()
        #self.avg_pooling = nn.AvgPool2d(kernel_size=pool_kernel)
        self.fc = nn.Linear(1024, out_classes)



    def forward(self, X):
        wav2vecout = self.encoder(X)
        context = wav2vecout.last_hidden_state
        pooled = torch.mean(context, dim=1)
        #vector = pooled.reshape(pooled.shape[0],pooled.shape[1]*pooled.shape[2])
        linear = self.fc(pooled)
        return linear

    def freeze_pretrained(self):
        '''freeze layers in pretrained model'''
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.frozen = True

    def unfreeze_pretrained(self):
        '''unfreeze layers in pretrained model'''
        print("unfreezing pretrained")
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.frozen = False
