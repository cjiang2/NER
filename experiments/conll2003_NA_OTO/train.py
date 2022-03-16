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
    'epochs': 100,
    'batch_size': 32,
    
    'lr': 1e-3,

    'save_dir': os.path.join(ROOT_DIR, 'checkpoints', 'NER_conll2003_normal')
}

if __name__ == "__main__":
    # ####################
    # Load CONLL 2003 dataset
    train_file = os.path.join(ROOT_DIR, 'data', 'conll2003', 'train.txt')
    valid_file = os.path.join(ROOT_DIR, 'data', 'conll2003', 'valid.txt')
    test_file = os.path.join(ROOT_DIR, 'data', 'conll2003', 'test.txt')

    all_tags = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
    tasks = [
        ['B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'],
        ['B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'],
        ['B-MISC', 'I-MISC'],
    ]

    #{'O': 13769, 'B-PER': 4373, 'I-PER': 2964, 'B-ORG': 4587, 'I-ORG': 2080, 'B-LOC': 5127, 'I-LOC': 949, 'B-MISC': 2698, 'I-MISC': 800}
    
    # {'O': 4135, 'B-PER': 1308, 'I-PER': 649, 'B-ORG': 0, 'I-ORG': 0, 'B-LOC': 0, 'I-LOC': 0, 'B-MISC': 0, 'I-MISC': 0}
    # {'O': 7126, 'B-PER': 1810, 'I-PER': 970, 'B-ORG': 3068, 'I-ORG': 1396, 'B-LOC': 0, 'I-LOC': 0, 'B-MISC': 0, 'I-MISC': 0}
    # {'O': 11162, 'B-PER': 3455, 'I-PER': 2373, 'B-ORG': 3910, 'I-ORG': 1735, 'B-LOC': 4058, 'I-LOC': 697, 'B-MISC': 0, 'I-MISC': 0}

    train_dataset = conll.NA_OTO_CONLL03(train_file, tags_to_remove=['B-MISC', 'I-MISC'])
    #exit()
    valid_dataset = conll.NA_OTO_CONLL03(test_file, vocab=train_dataset.vocab, tags_to_remove=['B-MISC', 'I-MISC'])
    train_loader = conll.get_loader(train_dataset, batch_size=config['batch_size'])
    valid_loader = conll.get_loader(valid_dataset, batch_size=1)

    # ####################
    # Model setup
    model = SimpleLSTM(vocab_size=len(train_dataset.vocab), num_classes=len(conll.NER_TAGS_CONLL03))
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    # ####################
    # Train
    trainer = BaselineTrainer(
        config=config,
        train_loader=train_loader,
        valid_loader=valid_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        )
    trainer.train()