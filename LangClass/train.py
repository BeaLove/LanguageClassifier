import numpy as np
import torch.nn

from wav2vecclassifier import LanguageClassifier
from dataloader import SentenceData
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from tqdm import tqdm

class Trainer():
    def __init__(self, data_dir):
        self.optim = Adam
        self.model = LanguageClassifier()
        dataset = SentenceData(data_dir)
        indices = np.arange(len(dataset.dataset))
        val_split = int(len(indices)*0.05)
        self.trainset = Subset(dataset, indices=indices[:val_split])
        self.val_set = Subset(dataset, indices=indices[val_split:])
        self.train_loader = DataLoader(self.trainset, shuffle=False, num_workers=4, batch_size=64)
        self.val_loader = DataLoader(self.val_set, shuffle=False, num_workers=4, batch_size=64)
        self.loss = torch.nn.CrossEntropyLoss()
        self.tensorboard_writer = torch.utils.tensorboard.SummaryWriter()


    def train_step(self):
        batch_i = tqdm(self.train_loader)
        '''debug code'''
        batch_i.set_description(desc="Training")
        for sample in batch_i:
            x, y = sample
            self.optim.zero_grad()
            output = self.model.forward(x)
            loss = self.loss(output)
            loss.backward()
            self.optim.step()

            metric = {"train loss: ": loss.cpu().detach().item()}

            batch_i.set_postfix(metric)

            self.tensorboard_writer.add_scalar(tag='train_loss', scalar_value=loss)






    def early_stop_callback(self):
        raise NotImplementedError

