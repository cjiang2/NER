"""
Baseline train script.

No CL method involved.
"""

import os
import sys
import shutil

import torch

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)  # To find local version of the library

from ner.data import conll
from ner.common import Vocabulary, maxlen
from ner.model.simple_lstm import SimpleLSTM
from ner.trainer.si import SITrainer
from ner.trainer.baseline import BaselineTrainer
from ner.common.word2vec import get_embed_matrix

config = {
    'epochs': 50,
    'batch_size': 32,
    'lr': 1e-3,

    'vocab_size': 23624,

    'save_dir': os.path.join(ROOT_DIR, 'checkpoints', 'NER_conll2003_seq_si'),

    'si_c': 0.1,
    'si_epsilon': 0.1,

    'word2vec': os.path.join(ROOT_DIR, 'checkpoints', 'GoogleNews-vectors-negative300.bin'),
    #'word2vec': None,
}

def main():
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
            ['B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'],
            ['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'],
            ['B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC'],
            ['B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC'],
        ],

    ]

    # ####################
    # Model setup
    model = SimpleLSTM(vocab_size=config['vocab_size'], num_classes=len(conll.NER_TAGS_CONLL03))   # Fixed vocab size
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    # Load a pretrained embedding matrix
    train_dataset = conll.NA_OTO_CONLL03(train_file, tags_to_remove=[], multiple_allowed=False)
    if config['word2vec'] is not None:
        vocab = train_dataset.vocab
        embed_mat = get_embed_matrix(config['word2vec'], 
                        embed_dim=300,
                        vocab=vocab)
        print(embed_mat.shape)
        model.embed.weight.data = torch.from_numpy(embed_mat).float()

    model.embed.requires_grad = False

    # Convenience: Do all experiments in one script
    for exp_i, tasks in enumerate(exps):
        f = open('exp_{}_si.txt'.format(exp_i + 1, exp_i), 'a')
        f.write("Tasks: {}\n".format(tasks))

        # Construct all datasets
        train_loaders = []
        valid_loaders = []
        for task in tasks:
            train_dataset = conll.NA_OTO_CONLL03(train_file, tags_to_remove=task, multiple_allowed=False)
            valid_dataset = conll.NA_OTO_CONLL03(test_file, vocab=train_dataset.vocab, tags_to_remove=task, multiple_allowed=False)
            train_loader = conll.get_loader(train_dataset, batch_size=config['batch_size'])
            valid_loader = conll.get_loader(valid_dataset, batch_size=1)
            train_loaders.append(train_loader)
            valid_loaders.append(valid_loader)

        # Optimizer reinitialized for every task
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'])
        trainer = SITrainer(
                name=None,
                config=config,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                si_c=config['si_c'],
                si_epsilon=config['si_epsilon'],
                )

        log_prev = []
        for task_idx, train_loader in enumerate(train_loaders):
            valid_loader = valid_loaders[task_idx]
            print("#"*30)
            print("Starting task:", task_idx + 1, tasks[task_idx])
            #print("#"*30)
            name = "exp_{}_task_{}".format(exp_i + 1, task_idx + 1)
            trainer.name = name

            # Write
            f.write("Task {}: trainset sentence counts: {}\n".format(task_idx + 1, train_loader.dataset.sentence_counts))
            print(train_loader.dataset.sentence_counts)

            # ####################
            # Train
            trainer.train(train_loader, valid_loader)

            # Reset
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['lr'])
            trainer.optimizer = optimizer
            trainer.log = {'f1': 0.0}

            # Load last best known model
            log = trainer.load_checkpoint(os.path.join(config['save_dir'], name+".pth"))
            log_prev.append(log['f1'])  # Save best performance for the previous task

            # Force a test against all valid_loaders
            log_new = []
            for k in range(task_idx + 1):
                print("Valid:", valid_loaders[k].dataset.sentence_counts, k)
                tester = BaselineTrainer(
                    name=name,
                    config=config,
                    train_loader=None,
                    valid_loader=valid_loaders[k],
                    model=trainer.model,
                    criterion=criterion,
                    optimizer=None,
                    )
                log = tester.valid_epoch(0)
                log_new.append(log['f1'])
            print("Prev:", log_prev)
            print("New:", log_new)

        for k, valid_loader in enumerate(valid_loaders):
            f.write("Task {}: validset sentence counts: {}\n".format(k + 1, valid_loader.dataset.sentence_counts))
        f.write("Prev Best Metric: {}\n".format(log_prev))
        f.write("Curr Best Metric: {}\n".format(log_new))
        f.close()

if __name__ == "__main__":
    for _ in range(1):
        if os.path.exists(config['save_dir']):
            shutil.rmtree(config['save_dir'])
        main()