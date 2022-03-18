"""
Baseline train script.

No CL method involved.
"""

from audioop import reverse
import os
import sys

import torch

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)  # To find local version of the library

from ner.data import conll
from ner.common import Vocabulary, maxlen
from ner.model.simple_lstm import SimpleLSTM
from ner.trainer.baseline import BaselineTrainer

config = {
    'epochs': 50,
    'batch_size': 32,
    'lr': 1e-3,

    'vocab_size': 23624,

    'save_dir': os.path.join(ROOT_DIR, 'checkpoints', 'NER_conll2003_na_normal'),

    'multiple_allowed': True,
}

if __name__ == "__main__":
    # ####################
    # Load CONLL 2003 dataset
    train_file = os.path.join(ROOT_DIR, 'data', 'conll2003', 'train.txt')
    valid_file = os.path.join(ROOT_DIR, 'data', 'conll2003', 'valid.txt')
    test_file = os.path.join(ROOT_DIR, 'data', 'conll2003', 'test.txt')

    all_tags = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

    exps = [
        [   # Experiment 1
            # Task 1: 
                # First train w/ samples w/o 'B-MISC', 'I-MISC' tags
            # Task 2:
                # Then train w/ samples w 'B-MISC', 'I-MISC' tags only
            'B-MISC', 'I-MISC',
        ],

        [   # Experiment 2: 'B-LOC', 'I-LOC'
            'B-LOC', 'I-LOC',
        ],

        [   # Experiment 3: 'B-ORG', 'I-ORG'
            'B-ORG', 'I-ORG',
        ],

        [   # Experiment 4: 'B-PER', 'I-PER'
            'B-PER', 'I-PER',
        ],

    ]

    # ####################
    # Model setup
    model = SimpleLSTM(vocab_size=config['vocab_size'], num_classes=len(conll.NER_TAGS_CONLL03))   # Fixed vocab size
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    # Convenience: Do all experiments in one script
    for exp_i, task in enumerate(exps):
        # Record
        if config["multiple_allowed"]:
            postfix = "MA"
        else:
            postfix = "OTO"
        f = open('exp_{}_{}.txt'.format(exp_i + 1, postfix), 'w')
        f.write("Task: {}\n".format(task))

        # New Addition, One-tag-Only
        for task_idx in range(2):
            print("#"*30)
            print("Starting task:", task_idx + 1)
            print("#"*30)
            name = "exp_{}_NA_{}_task_{}".format(exp_i + 1, postfix, task_idx + 1)

            if task_idx >= 1:
                reversed = True
            else:
                reversed = False

            train_dataset = conll.NA_OTO_CONLL03(train_file, tags_to_remove=task, multiple_allowed=config['multiple_allowed'], reversed=reversed)
            valid_dataset = conll.NA_OTO_CONLL03(test_file, vocab=train_dataset.vocab, tags_to_remove=task, multiple_allowed=config['multiple_allowed'], reversed=reversed)
            train_loader = conll.get_loader(train_dataset, batch_size=config['batch_size'])
            valid_loader = conll.get_loader(valid_dataset, batch_size=1)
            print(len(train_dataset.vocab))

            # Write
            f.write("Task {}: trainset sentence counts: {}\n".format(task_idx + 1, train_dataset.sentence_counts))
            f.write("Task {}: validset sentence counts: {}\n".format(task_idx + 1, valid_dataset.sentence_counts))

            # Optimizer needs to be reinitialized for every task, w/o CL trainer
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

            # Load the last known best model, if available
            if os.path.exists(os.path.join(config['save_dir'], name + '.pth')):
                checkpoint = torch.load(os.path.join(config['save_dir'], name + '.pth'))
                model.load_state_dict(checkpoint['state_dict'])
                print("[!] Used best weights from prev task {}".format(task_idx))
    
            # ####################
            # Train
            # Create a new trainer for each task, if using a baseline training method
            trainer = BaselineTrainer(
                name=name,
                config=config,
                train_loader=train_loader,
                valid_loader=valid_loader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                )
            trainer.train()

        # Re-evaluate preformance on previous task
        name = "exp_{}_NA_{}_task_{}".format(exp_i + 1, postfix, task_idx + 1)
        checkpoint = torch.load(os.path.join(config['save_dir'], name + '.pth'))
        model.load_state_dict(checkpoint['state_dict'])
        print("[!] Loaded latest task {}'s best weight.".format(task_idx + 1))
        valid_dataset = conll.NA_OTO_CONLL03(test_file, vocab=train_dataset.vocab, tags_to_remove=task, multiple_allowed=config['multiple_allowed'])
        valid_loader = conll.get_loader(valid_dataset, batch_size=1)

        # Lazy Way: just do another trainer and force using valid_epoch
        trainer = BaselineTrainer(
                name=name,
                config=config,
                train_loader=None,
                valid_loader=valid_loader,
                model=model,
                criterion=criterion,
                optimizer=None,
                )
        log = trainer.valid_epoch(0)

        name = "exp_{}_NA_{}_task_{}".format(exp_i + 1, postfix, task_idx)
        checkpoint = torch.load(os.path.join(config['save_dir'], name + '.pth'))
        log_prev = checkpoint['log']
        print("Prev Performance:", log_prev)
        print("New Performance:", log)

        f.write("Prev Best Metric: {}\n".format(log_prev))
        f.write("Curr Best Metric: {}\n".format(log))
        f.close()