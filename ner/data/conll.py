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

NER_TAGS_CONLL03_CLASSES = [
    'O', 
    'B-PER', 
    'I-PER', 
    'B-ORG', 
    'I-ORG', 
    'B-LOC', 
    'I-LOC', 
    'B-MISC', 
    'I-MISC',
]


# ##########

def load_data(
    filename: str, 
    lower_case: bool = True):
    """Read dataset in CoNLL 2003 format. Collect all tagging annotations, including NER.
    Reference:
        https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Sequence-Labeling/blob/master/utils.py#L12
    Returns:
        sentences: raw text data.
        tags: annotated tags.
    """
    sentences, tags = [], []

    # Temp
    guid = 0
    tokens = []
    ner_tags = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            # End of last sample
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if tokens:
                    if len(tokens) > 0:
                        sentences.append(tokens)
                        tags.append(ner_tags)
                        guid += 1
                        tokens = []
                        ner_tags = []

            # Sample at the end, append
            else:
                splits = line.split(" ")
                tokens.append(splits[0])
                ner_tags.append(splits[3].rstrip())

        # last sample
        if len(tokens) > 0:
            sentences.append(tokens)
            tags.append(ner_tags)
            guid += 1

    # Sanity check
    assert len(sentences) == len(tags)

    return sentences, tags


class CONLL03(Dataset):
    def __init__(
        self,
        filename: str,
        vocab: object = None,
        ):
        self.sentences, self.labels = load_data(filename)

        if vocab is None:
            self.vocab = Vocabulary()
            self.vocab.construct(self.sentences)
        else:
            self.vocab = vocab

        self.class_counts = self.class_counts_individual(self.labels)

    def class_counts_individual(self, labels):
        class_counts = [0 for _ in range(len(NER_TAGS_CONLL03))]
        for label in labels:
            for tag in label:
                class_counts[NER_TAGS_CONLL03[tag]] += 1
        return class_counts

    def __getitem__(self, i):
        # word2idx
        sentence = self.sentences[i]
        x = self.vocab.text_to_id(sentence)

        # tag2idx
        tags = self.labels[i]
        label = []
        for tag in tags:
            label.append(NER_TAGS_CONLL03[tag])

        return torch.Tensor(x), torch.Tensor(label)

    def __len__(self):
        return len(self.sentences)


class NA_MA_CONLL03(Dataset):
    """New addition, multiple allowed.
    """
    def __init__(
        self,
        filename: str,
        tags_to_remove: list,
        vocab: object = None,
        ):
        sentences, labels = load_data(filename)

        if vocab is None:
            self.vocab = Vocabulary()
            self.vocab.construct(sentences)
        else:
            self.vocab = vocab

        # Do some class summary here
        labels_unique = {}
        for label in labels:
            tags = ', '.join(sorted(set(label)))

            if tags not in labels_unique:
                labels_unique[tags] = 1
            else:
                labels_unique[tags] += 1
        
        self.class_counts_individual(labels)

        self.sentences, self.labels = [], []
        for i, sentence in enumerate(sentences):
            label = labels[i]
            collect = True
            for tag in tags_to_remove:
                if tag in label:
                    collect = False
            if collect:
                self.sentences.append(sentence)
                self.labels.append(label)

        self.class_counts_individual(self.labels)

    def class_counts_individual(self, labels):
        class_counts = {t: 0 for t in NER_TAGS_CONLL03}
        for label in labels:
            tags = sorted(set(label))
            for tag in tags:
                class_counts[tag] += 1
        print(class_counts)

    def __getitem__(self, i):
        # word2idx
        sentence = self.sentences[i]
        x = self.vocab.text_to_id(sentence)

        # tag2idx
        tags = self.labels[i]
        label = []
        for tag in tags:
            label.append(NER_TAGS_CONLL03[tag])

        return torch.Tensor(x), torch.Tensor(label)

    def __len__(self):
        return len(self.sentences)


class NA_OTO_CONLL03(Dataset):
    """New addition, one tag only.
    """
    def __init__(
        self,
        filename: str,
        tags_to_remove: list,
        vocab: object = None,
        ):
        sentences, labels = load_data(filename)
        sentences_one, labels_one = [], []

        # Keep only one tag per (tag, O) sentence
        for i, label in enumerate(labels):
            tags = sorted(set(label))
            if len(tags) <= 2:
                sentences_one.append(sentences[i])
                labels_one.append(label)

        if vocab is None:
            self.vocab = Vocabulary()
            self.vocab.construct(sentences)     # Still use full dataset to construct vocab
        else:
            self.vocab = vocab
        

        self.class_counts_individual(labels_one)

        self.sentences, self.labels = [], []
        for i, sentence in enumerate(sentences_one):
            label = labels_one[i]
            collect = True
            for tag in tags_to_remove:
                if tag in label:
                    collect = False
            if collect:
                self.sentences.append(sentence)
                self.labels.append(label)

        self.class_counts_individual(self.labels)

    def class_counts_individual(self, labels):
        class_counts = {t: 0 for t in NER_TAGS_CONLL03}
        for label in labels:
            tags = sorted(set(label))
            for tag in tags:
                class_counts[tag] += 1
        print(class_counts)

    def __getitem__(self, i):
        # word2idx
        sentence = self.sentences[i]
        x = self.vocab.text_to_id(sentence)

        # tag2idx
        tags = self.labels[i]
        label = []
        for tag in tags:
            label.append(NER_TAGS_CONLL03[tag])

        return torch.Tensor(x), torch.Tensor(label)

    def __len__(self):
        return len(self.sentences)


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
    targets = -1 * torch.ones(len(labels), max(lengths)).long()

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