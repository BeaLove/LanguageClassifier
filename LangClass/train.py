import torch.nn
import argparse
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
    def __init__(self, data_dir, log_dir, patience, checkpoints_dir, checkpoint):
        if checkpoint is not None:
            self.model = torch.load(checkpoint)
        else:
            self.model = LanguageClassifier()
        self.optim = torch.optim.Adam(self.model.parameters())
        self.dataset = SentenceData(data_dir)
        indices = torch.randperm(len(self.dataset))
        val_split = int((len(indices)*0.05))
        self.val_set = Subset(self.dataset, indices=indices[:val_split])
        self.trainset = Subset(self.dataset, indices=indices[val_split:])
        self.train_loader = DataLoader(self.trainset, shuffle=True, num_workers=3, batch_size=1)
        self.val_loader = DataLoader(self.val_set, shuffle=True, num_workers=3, batch_size=1)
        self.loss_criterion = torch.nn.CrossEntropyLoss()
        self.tensorboard_writer = torch.utils.tensorboard.SummaryWriter(log_dir=log_dir)
        self.global_loss = 1000
        self.patience = patience
        os.makedirs(checkpoints_dir, exist_ok=True)
        self.checkpt_dir = checkpoints_dir
        self.best_model = 0
        self.avg_val_loss = 0
        self.avg_train_loss = 0
        if torch.cuda.is_available():
            self.device = 'cuda:0'
            print("using cuda")
        else:
            self.device = 'cpu'
            print('using cpu!')
        self.model = self.model.to(self.device)
        print(self.model)
        print("dataset size: training {}, validation {}".format(len(self.trainset), len(self.val_set)))

    def train_epoch(self, epoch):
        batch_i = tqdm(self.train_loader)
        batch_i.set_description(desc="Training")
        sum_loss = 0
        for step, sample in enumerate(batch_i):
            x, y = sample
            x = x.to(self.device)
            y = y.to(self.device)
            x = x[0,:,:]
            self.optim.zero_grad()
            output = self.model.forward(x)
            loss = self.loss_criterion(output, y)
            sum_loss += loss.cpu().detach().item()
            loss.backward()
            self.optim.step()
            metric = {"train loss: ": loss, "Average train loss: ": self.avg_train_loss}
            batch_i.set_postfix(metric)
            self.tensorboard_writer.add_scalar(tag='train_loss', scalar_value=loss, global_step=epoch*step)
            #self.tensorboard_writer.add_scalar(tag="learning rate", scalar_value=self.optim)
            self.tensorboard_writer.add_histogram(tag="fc weight", values=self.model.fc.weight, global_step=epoch*step)
            self.tensorboard_writer.add_histogram(tag='fc layer bias', values=self.model.fc.bias, global_step=epoch*step)
        avg_train_loss = sum_loss/len(self.trainset)
    def validate(self, epoch):
        batch_i = tqdm(self.val_loader)
        batch_i.set_description(desc="validating")
        sum_loss = 0
        with torch.no_grad():
            for step, sample in enumerate(batch_i):
                x, y = sample
                x = x.to(self.device)
                y = y.to(self.device)
                x = x[0, :, :]
                output = self.model.forward(x)
                loss = self.loss_criterion(output, y)
                sum_loss += loss.cpu().detach().item()
                metric = {"running validation loss: ": loss.cpu().detach().item(), "avg val loss": self.avg_val_loss}
                batch_i.set_postfix(metric)
            self.avg_val_loss = sum_loss/len(self.val_set)
        self.tensorboard_writer.add_scalar(tag="val loss", scalar_value=self.avg_val_loss, global_step=epoch)
        return self.avg_val_loss


    def train(self, epochs):
        for epoch in range(1, epochs):
            self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            chkpt_name = 'wav2vec_finetune_checkpoint_epoch{}.pt'.format(epoch)
            torch.save(self.model, os.path.join(self.checkpt_dir, chkpt_name))
            if self.early_stop_callback(val_loss, epoch):
                print("Validation loss did not improve for {} epochs, stopping training!")


    def early_stop_callback(self, loss, epoch):
        if loss > self.global_loss:
            self.patience -= 1
            print("val loss did not improve, decreasing patience to: {}".format(self.patience))
        else:
            self.best_model = epoch
            self.global_loss = loss
            print("val loss improved!")
        if self.patience == 0:
            return True
        return False

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a Language Classifier on Wav2Vec embeddings")
    parser.add_argument('--dataset_dir', dest='dataset_dir', help='dataset directory', type=str)
    parser.add_argument('--checkpoints_dir', dest='ckpt_dir', default='checkpoints', type=str)
    parser.add_argument('--log_dir', dest='log_dir', default='logs', type=str)
    parser.add_argument('--epochs', dest='epochs', help='training epochs', type=int)
    parser.add_argument('--patience', dest='patience', help='early stop patience', default=5, type=int)
    parser.add_argument('--train_from', dest='train_from', default=None, help='resume training from checkpoint', type=str)
    '''currently not used'''
    parser.add_argument('-learning_rate', dest='lr', default=1e-2, type=float)
    parser.add_argument('--warm_up', dest='warm_up', default=0, type=int, help='number of warmup steps for optimizer')
    parser.add_argument('--decay_steps', dest='decay', default=0, type=int, help='number of steps to decay learning rate' )
    args = parser.parse_args()
    print(args)
    return args

if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(data_dir=args.dataset_dir, checkpoint=args.train_from, checkpoints_dir=args.ckpt_dir, log_dir=args.log_dir, patience=args.patience)
    trainer.train(epochs=args.epochs)

