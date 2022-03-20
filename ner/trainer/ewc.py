"""
"Standard" trainer with cross entropy.

"""

import random

import torch
import torch.nn.functional as F
import numpy as np 
from seqeval.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

from . import BaseTrainer
from ..utils import Progbar
from ..data.conll import NER_TAGS_CONLL03_CLASSES, get_loader

torch.backends.cudnn.enabled = False

class EWCTrainer(BaseTrainer):
    def __init__(
        self,
        name,
        config,
        model,
        criterion,
        optimizer,
        n_fisher_sample = 512,
        lambd = 200,
        ):
        super().__init__(name, config)
        self.model = model.cuda()
        self.criterion = criterion.cuda()
        self.optimizer = optimizer
        self.n_fisher_sample = n_fisher_sample
        self.lambd = lambd

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        self.log = {'f1': 0.0}

        # Initialization
        self.params_name = [n for n, p in self.model.named_parameters() if p.requires_grad]
        self.prev_params = {}
        self.fisher = {}
        for n, p in self.model.named_parameters():
            if n in self.params_name:
                self.prev_params[n] = p.clone().detach()
                self.fisher[n] = p.clone().detach().fill_(0)  # zero initialized
            
    def train_step(
        self,
        batch,
        step: int,
        ):
        """Train step for NER task.
        """
        if torch.cuda.is_available():
            x = batch[0].cuda()
            y = batch[1].cuda()
        lens = batch[2]
        #print(x.shape)
        output = self.model(x, lens)
        loss = self.criterion(output, y)

        # EWC penalty
        penalty = 0.0
        for n, p in self.model.named_parameters():
            if n in self.params_name:
                _penalty = self.fisher[n] * (p - self.prev_params[n]) ** 2
                penalty += _penalty.sum()

        loss += self.lambd * penalty

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss}

    def train_epoch(
        self,
        epoch: int,
        train_loader,
        valid_loader,
        ):
        # Setup progbar
        pbar = Progbar(target=len(train_loader), 
                epoch=epoch, 
                num_epochs=self.config['epochs'])

        self.model.train()
        for batch_idx, batch in enumerate(train_loader):
            result = self.train_step(batch, batch_idx)
            pbar.update(batch_idx + 1, values=[('loss', result['loss'])])

        if valid_loader is not None:
            log = self.valid_epoch(epoch, valid_loader)

        # Save best valid f1
        self.save_best_checkpoint(epoch, log, self.config['save_dir'])

        return log

    def train(
        self,
        train_loader,
        valid_loader,
        ):
        """Generic training loop.
        """
        for epoch in range(self.start_epoch, self.config["epochs"] + 1):
            log = self.train_epoch(epoch, train_loader, valid_loader)
        print('Training done.')

    def update_fisher(self, sub_loader):
        print("prev:", self.prev_params['fc.weight'][:10])
        print(len(sub_loader.dataset))

        for n, p in self.model.named_parameters():
            if n in self.params_name:
                self.prev_params[n] = p.clone().detach()

        self.model.eval()
        for i, batch in enumerate(sub_loader):
            self.model.zero_grad()
            if torch.cuda.is_available():
                x = batch[0].cuda()
                y = batch[1].cuda()
            lens = batch[2]
            output = self.model(x, lens)
            loss = F.nll_loss(F.log_softmax(output, dim=1), y, ignore_index=-1)
            #loss = self.criterion(output, y)
            loss.backward()
            for n, p in self.model.named_parameters():
                if n in self.params_name:
                    self.fisher[n] += p.grad.data ** 2 / len(sub_loader.dataset)
        print("updated:", self.prev_params['fc.weight'][:10])
        print("fisher updated.")

    def on_update(self, train_loader):
        """Subsample dataset and calculate fisher matrix.
        """
        print(len(train_loader.dataset))
        n_sample = min(self.n_fisher_sample, len(train_loader.dataset))
        
        rand_ind = random.sample(list(range(len(train_loader.dataset))), n_sample)
        sub_dataset = torch.utils.data.Subset(train_loader.dataset, rand_ind)
        sub_loader = get_loader(sub_dataset, shuffle=True, batch_size=1)
        self.update_fisher(sub_loader)

    def save_best_checkpoint(
        self, 
        epoch: int,
        log_epoch: dict,
        save_dir: str,
        ):
        """Save best training model.
        """
        #if log_epoch['f1'] >= self.log['f1']:
        self.log = log_epoch
        filename = "{}.pth".format(self.name)
        self.save_checkpoint(epoch, save_dir, filename)

    def load_checkpoint(
        self,
        filepath: str,
        train_loader_prev,
        ):
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['state_dict'])
        print("Loaded previous best model. Updating fisher.")

        self.on_update(train_loader_prev)

        return checkpoint['log']

    def valid_epoch(
        self,
        epoch: int,
        valid_loader,
        ):
        print("\n Eval...")
        self.model.eval()
        preds, gts = [], []

        with torch.no_grad():
            for idx, batch in tqdm(enumerate(valid_loader)):
                if torch.cuda.is_available():
                    x = batch[0].cuda()
                y = batch[1]
                lens = batch[2]
                output = self.model(x, lens)
                probs = F.softmax(output, dim=1)
                pred = torch.argmax(probs, dim=1)

                pred = pred.cpu().numpy()[0].tolist()
                gt = y.numpy()[0].tolist()
                pred = [NER_TAGS_CONLL03_CLASSES[class_idx] for class_idx in pred]
                gt = [NER_TAGS_CONLL03_CLASSES[class_idx] for class_idx in gt]

                preds.append(pred)
                gts.append(gt)

        f1 = f1_score(gts, preds)
        print("Epoch: {}, F1: {:.6f}\n".format(epoch, f1))

        return {'f1': f1}