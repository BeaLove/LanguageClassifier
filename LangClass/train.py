import torch.nn

from wav2vecclassifier import LanguageClassifier
from dataloader import SentenceData
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Trainer():
    def __init__(self, data_dir):

        self.model = LanguageClassifier()
        self.optim = torch.optim.SGD(self.model.parameters(), lr=1e-2)
        self.dataset = SentenceData(data_dir)
        indices = torch.randperm(len(self.dataset))
        val_split = int((len(indices)*0.05))
        self.val_set = Subset(self.dataset, indices=indices[:val_split])
        self.trainset = Subset(self.dataset, indices=indices[500:]) ##TODO change when doing full run
        self.train_loader = DataLoader(self.trainset, shuffle=True, num_workers=0, batch_size=1)
        self.val_loader = DataLoader(self.val_set, shuffle=True, num_workers=0, batch_size=1)
        self.loss_criterion = torch.nn.CrossEntropyLoss()
        self.tensorboard_writer = torch.utils.tensorboard.SummaryWriter(log_dir='logs')
        self.global_loss = 1000
        self.patience = 5
        os.makedirs('checkpoints', exist_ok=True)
        self.checkpt_dir = 'checkpoints'
        self.best_model = 0
        if torch.cuda.is_available():
            self.device = 'cuda:0'
            print("using cuda")
        else:
            self.device = 'cpu'
            print('using cpu!')
        self.model = self.model.to(self.device)

    def train_epoch(self, epoch):
        batch_i = tqdm(self.train_loader)
        batch_i.set_description(desc="Training")
        for step, sample in enumerate(batch_i):
            x, y = sample
            x = x.to(self.device)
            y = y.to(self.device)
            x = x[0,:,:]
            self.optim.zero_grad()
            output = self.model.forward(x)
            loss = self.loss_criterion(output, y)
            loss.backward()
            self.optim.step()

            metric = {"train loss: ": loss.cpu().detach().item()}

            batch_i.set_postfix(metric)

            self.tensorboard_writer.add_scalar(tag='train_loss', scalar_value=loss, global_step=epoch*step)
            self.tensorboard_writer.add_histogram(tag="fc weight", values=self.model.fc.weight, global_step=epoch*step)
            self.tensorboard_writer.add_histogram(tag='fc layer bias', values=self.model.fc.bias, global_step=epoch*step)

    def validate(self, epoch):
        batch_i = tqdm(self.val_loader)
        batch_i.set_description(desc="validating")
        sum_loss = 0
        with torch.no_grad():
            for step, sample in enumerate(batch_i):
                x, y = sample
                output = self.model.forward(x)
                sum_loss += self.loss(output, y)
            loss = sum_loss/len(self.val_set)
        self.tensorboard_writer.add_scalar(tag="val loss", scalar_value=loss, global_step=epoch)
        return loss


    def train(self, epochs):
        for epoch in range(1, epochs):
            self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            chkpt_name = 'wav2vec_finetune_checkpoint_epoch{}.pt'.format(epoch)
            self.model.save(os.path.join(self.checkpt_dir, chkpt_name))
            if self.early_stop_callback(val_loss, epoch):
                print("Validation loss did not improve for {} epochs, stopping training!")


    def early_stop_callback(self, loss, epoch):
        if loss > self.global_loss:
            self.patience -= 1
        else:
            self.best_model = epoch
        if self.patience == 0:
            return True
        return False

trainer = Trainer(data_dir='data/train')
trainer.train(epochs=50)

