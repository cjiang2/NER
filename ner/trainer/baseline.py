"""
"Standard" trainer with cross entropy.

"""

import torch
import torch.nn.functional as F
import numpy as np 
from seqeval.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

from . import BaseTrainer
from ..utils import Progbar
from ..data.conll import NER_TAGS_CONLL03_CLASSES

class BaselineTrainer(BaseTrainer):
    def __init__(
        self,
        name,
        config,
        model,
        criterion,
        optimizer,
        ):
        super().__init__(name, config)
        self.model = model.cuda()
        self.criterion = criterion.cuda()
        self.optimizer = optimizer

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        self.log = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

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
        ):
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['state_dict'])
        print("Loaded previous best model.")

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