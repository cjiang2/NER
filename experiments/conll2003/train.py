import os
import sys

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)  # To find local version of the library

from ner.data import conll
from ner.common import Vocabulary, maxlen

if __name__ == "__main__":
    train_file = os.path.join(ROOT_DIR, 'data', 'conll2003', 'eng.train')
    testa_file = os.path.join(ROOT_DIR, 'data', 'conll2003', 'eng.testa')
    testb_file = os.path.join(ROOT_DIR, 'data', 'conll2003', 'eng.testb')

    train_dataset = conll.CONLL03(train_file)
    test_dataset_a = conll.CONLL03(testa_file, vocab=train_dataset.vocab)
    test_dataset_b = conll.CONLL03(testb_file, vocab=train_dataset.vocab)

    for i, batch in enumerate(test_dataset_b):
        print(batch[0])
        print(batch[1])
        print()

        #if i == 4:
        #    break