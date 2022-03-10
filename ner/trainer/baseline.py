"""
"Standard" trainer with cross entropy.

"""

import torch

from . import BaseTrainer
from ..utils import Progbar

class BaselineTrainer(BaseTrainer):
    def __init__(
        self,
        config,
        train_loader,
        valid_loader,
        model,
        criterion,
        optimizer,
        ):
        super().__init__(config)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model.cuda()
        self.criterion = criterion.cuda()
        self.optimizer = optimizer

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        self.log = {}

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
        ):
        # Setup progbar
        pbar = Progbar(target=len(self.train_loader), 
                epoch=epoch, 
                num_epochs=self.config['epochs'])

        self.model.train()
        for batch_idx, batch in enumerate(self.train_loader):
            result = self.train_step(batch, batch_idx)
            pbar.update(batch_idx + 1, values=[('loss', result['loss'])])

        if self.valid_loader is not None:
            log = self.valid_epoch(epoch)

        return log

    def valid_epoch(
        self,
        epoch: int,
        ):
        return {'acc': 0.0}