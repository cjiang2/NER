import os

import numpy as np
import torch
from torch.utils.data import Dataset

from ..common import Vocabulary

# ##########
# For Specific Datasets
# ##########

TAGS_CONLL03 = [
    "POS",
    "chunk",
    "NER",
]

NER_TAGS_CONLL03 = {
    'O': 0, 
    'B-PER': 1, 
    'I-PER': 2, 
    'B-ORG': 3, 
    'I-ORG': 4, 
    'B-LOC': 5, 
    'I-LOC': 6, 
    'B-MISC': 7, 
    'I-MISC': 8,
}


# ##########

def load_data(
    filename: str, 
    tags_to_add: list = TAGS_CONLL03, 
    lower_case: bool = True):
    """Read dataset in CoNLL 2003 format. Collect all tagging annotations, including NER.
    Reference:
        https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Sequence-Labeling/blob/master/utils.py#L12
    Returns:
        words: raw text data.
        tags: annotated tags.
    """
    words, tags = [], []
    word_sample = []
    tag_sample = {}
    for tag in tags_to_add:
        tag_sample[tag] = []

    with open(filename, encoding='utf-8') as f:
        for line in f:
            # Ignore "DOCSTART"
            if not (line.isspace() or (len(line) > 10 and line[0:10] == '-DOCSTART-')):
                line = line.rstrip('\n').split()
                
                word_sample.append(line[0].lower() if lower_case else line[0])
                for i, tag in enumerate(tags_to_add):
                    tag_sample[tag].append(line[i + 1])

            # Sample at the end, append
            elif len(word_sample) > 0:
                for tag in tags_to_add: assert len(word_sample) == len(tag_sample[tag])
                words.append(word_sample)
                tags.append(tag_sample)
                word_sample = []
                tag_sample = {}
                for tag in tags_to_add:
                    tag_sample[tag] = []

        # last sentence
        if len(word_sample) > 0:
            for tag in tags_to_add: assert len(word_sample) == len(tag_sample[tag])
            words.append(word_sample)
            tags.append(tag_sample)

    # Sanity check
    assert len(words) == len(tags)

    return words, tags


class CONLL03(Dataset):
    def __init__(
        self,
        filename: str,
        task: str = "NER",
        vocab: object = None,
        ):
        self.task = task

        self.TAG_LABELS = NER_TAGS_CONLL03

        self.words, self.tags = load_data(filename, tags_to_add=TAGS_CONLL03)

        if vocab is None:
            self.vocab = Vocabulary()
            self.vocab.construct(self.words)
        else:
            self.vocab = vocab

    def __getitem__(self, i):
        # word2idx
        word = self.words[i]
        x = self.vocab.text_to_id(word)

        # tag2idx
        tags = self.tags[i][self.task]
        label = []
        for tag in tags:
            label.append(self.TAG_LABELS[tag])

        print(word, len(word))
        print(tags, len(tags))
        #assert len(x) == len(label)
        assert -1 not in tags
        return torch.Tensor(x), torch.Tensor(label)

    def __len__(self):
        return len(self.words)


def collate_fn(data):
    """Build mini-batch tensors from a list of (texts, labels) tuples.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[0]), reverse=True)
    X, labels = zip(*data)

    # lengths
    lengths = [len(label) for label in labels]

    # Merge (convert tuple of 1D tensor to 2D tensor)
    inputs = torch.zeros(len(X), max(lengths)).long()
    targets = torch.zeros(len(labels), max(lengths)).long()

    for i in range(len(X)):
        end = lengths[i]
        inputs[i, :end] = X[i][:end]
        targets[i, :end] = labels[i][:end]

    return inputs, targets, lengths

def get_loader(
    dataset, 
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    collate_fn=collate_fn,
    ):
    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader