import os
import sys

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)  # To find local version of the library

from ner.data import conll

if __name__ == "__main__":
    train_file = os.path.join(ROOT_DIR, 'data', 'conll2003', 'eng.train')
    testa_file = os.path.join(ROOT_DIR, 'data', 'conll2003', 'eng.testa')
    testb_file = os.path.join(ROOT_DIR, 'data', 'conll2003', 'eng.testb')

    words_train, tags_train = conll.load_data(train_file, 1)
    print(len(words_train))
    print(words_train[0], tags_train[0])

    words_testa, tags_testa = conll.load_data(testa_file, 1)
    words_testb, tags_testb = conll.load_data(testb_file, 1)
    print(len(words_testa))
    print(words_testa[0], tags_testa[0])