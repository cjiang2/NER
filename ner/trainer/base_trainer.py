"""
Base trainer class.

"""
import os
import datetime
from abc import abstractmethod

import torch

class BaseTrainer:
    """Base class for all trainers.
    """
    def __init__(
        self,
        name: str,
        config,
        ) -> None:
        self.name = name
        self.config = config
        
        self.start_epoch = 1
        self.log = {}
        self.start_time = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    @abstractmethod
    def train_epoch(
        self, 
        epoch: int,
        ):
        """Training logic for one epoch.
        Args:
            epoch: current epoch no.
        """
        raise NotImplementedError

    @abstractmethod
    def train_step(
        self, 
        batch,
        step: int,
        ):
        """Training logic for one step.
        Args:
            step: current step no.
        """
        raise NotImplementedError

    @abstractmethod
    def train(
        self,
        ):
        raise NotImplementedError

    def save_checkpoint(
        self, 
        epoch: int,
        save_dir: str,
        filename: str = None,
        ):
        """Save best checkpoint if possible.
        Invoke _save_checkpoint here.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        state = {
            'epoch': epoch,
            'log': self.log,
            'state_dict': self.model.state_dict(),
        }
        if filename is None:
            filename = "model_epoch_{}.pth".format(epoch)
        torch.save(state, os.path.join(save_dir, filename))
        print('[!] Model saved at epoch {}.'.format(epoch))

    @abstractmethod
    def save_best_checkpoint(
        self, 
        epoch: int,
        log_epoch: dict,
        save_dir: str,
        ):
        """Save best training model.
        """
        raise NotImplementedError