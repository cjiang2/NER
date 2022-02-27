import os

import numpy as np

# ##########
# For Specific Datasets
# ##########

TAGS_CONLL03 = [
    "POS",
    "chunk",
    "NER",
]


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
                    tag_sample[tag].append(line[i])

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


