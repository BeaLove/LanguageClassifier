import torch.nn
import argparse
from wav2vecclassifier import LanguageClassifier
from dataloader import Commonvoice, VoxLingua
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Trainer():
    def __init__(self, data_dir, log_dir, batch_size, patience, checkpoints_dir, checkpoint,
                 lr, max_lr, min_lr, optim, warmup_steps, decay_steps, unfreeze_after):
        '''initialize training with options:
            -start training from checkpoint or from scratch
            -define path to data, tensorboard log directory, checkpoints directory, learning rate'''
        if checkpoint is not None:
            print("training from: ", checkpoint)
            self.model = torch.load(checkpoint)
            self.model.unfreeze_pretrained()
        else:
            self.model = LanguageClassifier()
        if optim == 'sgd':
            self.optim = torch.optim.SGD(self.model.parameters(), lr=lr)
        elif optim == 'adam':
            self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        print("optimizer: ", self.optim)
        self.lr = lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        if warmup_steps != 0:
            self.use_warmup = True
            self.warmup_steps = warmup_steps
        else:
            self.warmup_steps = None
            self.use_warmup = False
        if decay_steps != 0:
            self.decay_steps = decay_steps
        else:
            self.decay_steps = None
        if unfreeze_after != 0:
            '''setting a custom step at which to unfreeze pretrained'''
            self.unfreeze_after = unfreeze_after
        self.dataset = Commonvoice(data_dir, sample_len=4) #instatiates dataset and split into training and validation sets (hardcoded 5% val data)
        indices = torch.randperm(len(self.dataset))
        val_split = int((len(indices)*0.05))
        self.val_set = Subset(self.dataset, indices=indices[:val_split])
        self.trainset = Subset(self.dataset, indices=indices[val_split:])
        self.train_loader = DataLoader(self.trainset, shuffle=True, num_workers=4, batch_size=batch_size)
        self.val_loader = DataLoader(self.val_set, shuffle=True, num_workers=4, batch_size=batch_size)
        self.loss_criterion = torch.nn.CrossEntropyLoss()
        self.tensorboard_writer = torch.utils.tensorboard.SummaryWriter(log_dir=log_dir)
        self.global_loss = 1000
        self.patience = patience
        self.max_patience = patience
        os.makedirs(checkpoints_dir, exist_ok=True)
        self.checkpt_dir = checkpoints_dir
        self.best_model = 0
        self.avg_val_loss = 0
        self.avg_train_loss = 0
        self.batch_size = batch_size
        if torch.cuda.is_available():
            self.device = 'cuda:0'
            print("using cuda")
        else:
            self.device = 'cpu'
            print('using cpu!')
        self.model = self.model.to(self.device)
        #print(self.model)
        print("dataset size: training {}, validation {}".format(len(self.trainset), len(self.val_set)))

    def train_epoch(self, epoch):
        '''runs one epoch of training (one full run through the training data'''
        dataset = tqdm(self.train_loader)
        dataset.set_description(desc="Training")
        sum_loss = 0
        batch = 0
        #batch_losses = torch.zeros(self.batch_size)
        total_losses = []

        for step, sample in enumerate(dataset):
            x, y = sample
            x = x.to(self.device)
            y = y.to(self.device)
            #x = x[0,:,:]
            output = self.model.forward(x)
            loss = self.loss_criterion(output, y)
            #batch_losses[batch] = loss
            total_losses.append(loss)
            #batch += 1
            #'''average loss over batch size training samples and perform backprop,
            #    this is needed because pre-trained wav2vec will only take one sample at a time, not batches'''
            #if batch == self.batch_size:
                #batch_losses.mean().backward()
            loss.backward()
            ##try gradient clipping
            torch.nn.utils.clip_grad_value_(parameters=self.model.parameters(), clip_value=0.5)
            self.optim.step()
            metric = {"epoch: ": epoch,
                      "smoothed loss ": loss.cpu().detach().item(),
                      "Average train loss: ": self.avg_train_loss}
            dataset.set_postfix(metric)
            '''log weights and gradients after update'''
            self.tensorboard_writer.add_scalar(tag='train loss', scalar_value=loss.cpu().detach().item(), global_step=epoch*step)
            self.tensorboard_writer.add_histogram(tag="fc weight", values=self.model.fc.weight,
                                                  global_step=epoch * step)
            self.tensorboard_writer.add_histogram(tag='fc layer bias', values=self.model.fc.bias,
                                                  global_step=epoch * step)
            self.tensorboard_writer.add_histogram(tag="fc layer weight grad", values=self.model.fc.weight.grad,
                                                  global_step=epoch * step)
            self.tensorboard_writer.add_histogram(tag="fc layer bias grad", values=self.model.fc.bias.grad,
                                                  global_step=epoch * step)
            #batch = 0
            #batch_losses = torch.zeros(self.batch_size)

            self.optim.zero_grad()
            if self.use_warmup and step <= self.warmup_steps:
                self.lr_rampup()
            elif self.use_warmup and step > self.warmup_steps:
                self.lr_decay()
            if self.model.frozen is True and step*epoch == self.unfreeze_after:
                '''unfreeze pretrained layer for last steps'''
                self.model.unfreeze_pretrained()
                #self.optim.add_param_group({'encoder': self.model.encoder})
            self.tensorboard_writer.add_scalar(tag='lr', scalar_value=self.lr, global_step=epoch*step)
            sum_loss += loss.cpu().detach().item()
        self.avg_train_loss = sum(total_losses)/len(total_losses)

    def validate(self, epoch):
        batch_i = tqdm(self.val_loader)
        batch_i.set_description(desc="validating")
        sum_loss = 0
        with torch.no_grad():
            for step, sample in enumerate(batch_i):
                x, y = sample
                x = x.to(self.device)
                y = y.to(self.device)
                #x = x[0, :, :]
                output = self.model.forward(x)
                loss = self.loss_criterion(output, y)
                sum_loss += loss.cpu().detach().item()
                metric = {"step-wise validation loss: ": loss.cpu().detach().item(), "avg val loss": self.avg_val_loss}
                batch_i.set_postfix(metric)
            self.avg_val_loss = sum_loss/len(self.val_set)
        self.tensorboard_writer.add_scalar(tag="val loss", scalar_value=self.avg_val_loss, global_step=epoch)
        return self.avg_val_loss

    def lr_rampup(self):
        '''implements linear learning rate warmup for given number of training steps'''
        self.lr = self.lr + self.max_lr *(1/self.warmup_steps)
        for group in self.optim.param_groups:
            group['lr'] = self.lr

    def lr_decay(self):
        '''decays learning rate linearly'''
        self.lr = self.lr - (self.max_lr-self.min_lr)*(1/self.decay_steps)
        for group in self.optim.param_groups:
            group['lr'] = self.lr

    def train(self, epochs):
        for epoch in range(1, epochs):
            self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            chkpt_name = 'wav2vec_finetune_epoch{}.pt'.format(epoch)
            torch.save(self.model, os.path.join(self.checkpt_dir, chkpt_name))
            if self.early_stop_callback(val_loss, epoch):
                print("Validation loss did not improve for {} epochs, stopping training!".format(self.patience))
                print("best model recorded at epoch {}, loss {}".format(self.best_model, self.global_loss))
                break

    def early_stop_callback(self, loss, epoch):
        if loss > self.global_loss:
            self.patience -= 1
            print("val loss did not improve, decreasing patience to: {}".format(self.patience))
        else:
            self.best_model = epoch
            self.global_loss = loss
            self.patience = self.max_patience
            print("val loss improved! best loss {} recorded at {}".format(self.global_loss, self.best_model))
        if self.patience == 0:
            return True
        return False

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a Language Classifier on Wav2Vec embeddings")
    parser.add_argument('--dataset_dir', dest='dataset_dir', default='data/train', help='dataset directory', type=str)
    parser.add_argument('--checkpoints_dir', dest='ckpt_dir', default='checkpoints', type=str)
    parser.add_argument('--log_dir', dest='log_dir', default='logs', type=str)
    parser.add_argument('--epochs', dest='epochs', default=50, help='training epochs', type=int)
    parser.add_argument('--patience', dest='patience', help='early stop patience', default=5, type=int)
    parser.add_argument('--train_from', dest='train_from', default=None, help='resume training from checkpoint', type=str)
    parser.add_argument('--learning_rate', dest='lr', default=1e-5, type=float, help='learning rate, default 1e-5, if warmup is enabled this is the starting lr')
    parser.add_argument('--optimizer', dest='optim', default='adam', type=str, help='which optimizer to use, default is Adam')
    parser.add_argument('--warmup', dest='warmup', default=150, type=int, help='number of warmup steps for optimizer, default 2500, set to 0 to disable warmup')
    parser.add_argument('--max_lr', dest='max_lr', default=5e-3, type=float, help='maximum learning rate attained after steps = warmup steps')
    parser.add_argument('--min_lr', dest='min_lr', default=1e-6, type=float, help='minimum learning rate from decay')
    parser.add_argument('--decay_steps', dest='decay', default=2000, type=int, help='number of steps to decay learning rate' )
    parser.add_argument('--unfreeze_after', dest='unfreeze_after', default=150, type=int, help="unfreeze pretrained layers after this number of steps, defalt = warmup steps")
    parser.add_argument('--batch_size', dest='batch_size', default=16, type=int, help='batch size')
    args = parser.parse_args()
    print(args)
    return args

if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(data_dir=args.dataset_dir, checkpoint=args.train_from, checkpoints_dir=args.ckpt_dir, lr=args.lr,
                      log_dir=args.log_dir, batch_size=args.batch_size, patience=args.patience, max_lr=args.max_lr, optim=args.optim, warmup_steps=args.warmup,
                      decay_steps=args.decay, unfreeze_after=args.unfreeze_after)
    trainer.train(epochs=args.epochs)

