
import torch.nn as nn
import torch
from transformers import Wav2Vec2ForCTC


class LanguageClassifier(nn.Module):
    def __init__(self, out_classes=3):
        '''instantiates a 3-language classifier for Swedish, English and Arabic'''
        super().__init__()
        self.encoder = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-large-xlsr-53')
        self.freeze_pretrained()
        self.frozen = True
        self.fc = nn.Linear(32, out_classes)



    def forward(self, X):
        wav2vecout = self.encoder(X)
        context = wav2vecout.logits
        pooled = torch.mean(context, dim=1)
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
