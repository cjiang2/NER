"""
Base trainer class.

"""
import os
import datetime
from abc import abstractmethod

class BaseTrainer:
    """Base class for all trainers.
    """
    def __init__(
        self,
        config,
        ) -> None:
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

    def train(
        self,
        ):
        """Generic training loop.
        """
        for epoch in range(self.start_epoch, self.config["epochs"] + 1):
            log = self.train_epoch(epoch)
        print('Training done.')

    @abstractmethod
    def save_checkpoint(
        self, 
        epoch: int,
        log: dict,
        ):
        """Save best checkpoint if possible.
        Invoke _save_checkpoint here.
        """
        raise NotImplementedError