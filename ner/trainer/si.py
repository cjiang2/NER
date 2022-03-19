"""
Trainer with Synatic Intelligence.

"""

import torch
import torch.nn.functional as F
import numpy as np 
from seqeval.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

from . import BaseTrainer
from ..utils import Progbar
from ..data.conll import NER_TAGS_CONLL03_CLASSES

class SITrainer(BaseTrainer):
    def __init__(
        self,
        name,
        config,
        model,
        criterion,
        optimizer,
        si_c: float = 0.1,
        si_epsilon: float = 0.1,
        ):
        super().__init__(name, config)
        # Base initialization
        self.model = model.cuda()
        self.criterion = criterion.cuda()
        self.optimizer = optimizer
        self.si_c = si_c
        self.si_epsilon = si_epsilon

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

        self.logs = None

        # SI
        self._si_init()
        
    
    def _si_init(self):
        # Setup required parameters for SI
        self.params_name = [n for n, p in self.model.named_parameters() if p.requires_grad]

        # Initialize importance measure w and omega as zeros, record previous weights
        self.w = {}
        self.omega = {}
        self.prev_params = {}
        for n, p in self.model.named_parameters():
            if n in self.params_name:
                self.prev_params[n] = p.clone().detach()
                self.w[n] = torch.zeros(p.shape).float().to(p.device)
                self.omega[n] = torch.zeros(p.shape).float().to(p.device)

    def train_step(
        self,
        batch,
        step: int,
        ):
        """Train step for NER task.
        """
        if torch.cuda.is_available():
            x = batch[0].cuda()
            target = batch[1].cuda()
        lens = batch[2]
        #print(x.shape)
        output = self.model(x, lens)

        # Collect unregularized gradients
        unreg_grads = {}
        unreg_loss = self.criterion(output, target)
        self.optimizer.zero_grad()
        unreg_loss.backward(retain_graph=True)
        for n, p in self.model.named_parameters():
            if n in self.params_name:
                unreg_grads[n] = p.grad.clone().detach()
        
        # Calculate surrogate loss
        surrogate_loss = 0.0
        for n, p in self.model.named_parameters():
            if n in self.params_name:
                surrogate_loss += torch.sum(self.omega[n]*((p.detach() - self.prev_params[n])**2))
        loss = self.criterion(output, target) + self.si_c * surrogate_loss

        # One train step with surrogate loss now
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update importance right after every train step
        for n, p in self.model.named_parameters():
            if n in self.params_name:
                delta = p.detach() - self.prev_params[n]
                self.w[n] = self.w[n] - unreg_grads[n]*delta

        return {'loss': loss}

    def on_update(self):
        # Calculate regularization strength
        for n, p in self.model.named_parameters():
            if n in self.params_name:
                self.omega[n] += self.w[n] / ((p.detach() - self.prev_params[n])**2 + self.si_epsilon)
                self.omega[n] = F.relu(self.omega[n])

                # Reset importance measure and record previous weights
                self.w[n] = self.w[n]*0.0
                self.prev_params[n] = p.clone().detach()

    def train_epoch(
        self,
        epoch: int,
        train_loader,
        valid_loaders,
        task_idx: int,
        ):
        # Setup progbar
        pbar = Progbar(target=len(train_loader), 
                epoch=epoch, 
                num_epochs=self.config['epochs'])

        self.model.train()
        for batch_idx, batch in enumerate(train_loader):
            result = self.train_step(batch, batch_idx)
            pbar.update(batch_idx + 1, values=[('loss', result['loss'])])

        # Valid
        if valid_loaders is not None:
            logs = self.valid_epoch(epoch, valid_loaders)

        # Save best valid f1
        self.save_best_checkpoint(epoch, logs, self.config['save_dir'], task_idx)

        return logs

    def train(
        self,
        train_loader,
        valid_loaders,
        task_idx: int,
        ):
        """Generic training loop.
        """
        for epoch in range(self.start_epoch, self.config["epochs"] + 1):
            log = self.train_epoch(epoch, train_loader, valid_loaders, task_idx)

        print('Training done. Updaing importance params...')

        # #####
        # SI
        # Update omega and the referene weights, set parameter importance to 0
        # #####
        self.on_update()
        print('Done.')


    def save_best_checkpoint(
        self, 
        epoch: int,
        logs: dict,
        save_dir: str,
        task_idx: int,
        ):
        """Save best training model.
        """
        if logs['f1'][task_idx] >= self.logs['f1'][task_idx]:
            self.logs = logs
            filename = "{}.pth".format(self.name)
            self.save_checkpoint(epoch, save_dir, filename)

    def valid_epoch(
        self,
        epoch: int,
        valid_loaders,
        ):
        print("\n Eval...")
        self.model.eval()
        preds, gts = [], []

        if self.logs is None:
            self.logs = {
                'f1': [0.0 for _ in range(len(valid_loaders))], 
                'precision': [0.0 for _ in range(len(valid_loaders))], 
                'recall': [0.0 for _ in range(len(valid_loaders))]}

        logs = {'f1': [], 'precision': [], 'recall': []}

        for task_idx, valid_loader in enumerate(valid_loaders):
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

            precision = precision_score(gts, preds)
            recall = recall_score(gts, preds)
            f1 = f1_score(gts, preds)
            print("Epoch: {}, Precision: {:.6f}, Recall: {:.6f}, F1: {:.6f}\n".format(epoch, precision, recall, f1))
            logs['f1'].append(f1)
            logs['precision'].append(precision)
            logs['recall'].append(recall)

        return logs