import os
import sys

import matplotlib.pyplot as plt
import torch

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)  # To find local version of the library

from ner.data import conll
from ner.common import Vocabulary, maxlen

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

    train_dataset = conll.NA_OTO_CONLL03(train_file, tags_to_remove=['B-PER', 'I-PER'])
    # exit()
    
    
    # stats = [{'O': 11162, 'B-PER': 3455, 'I-PER': 2373, 'B-ORG': 3910, 'I-ORG': 1735, 'B-LOC': 4058, 'I-LOC': 697, 'B-MISC': 0, 'I-MISC': 0},
    #          {'O': 8664, 'B-PER': 2260, 'I-PER': 1233, 'B-ORG': 3454, 'I-ORG': 1591, 'B-LOC': 0, 'I-LOC': 0, 'B-MISC': 1629, 'I-MISC': 523},
    #          {'O': 9259, 'B-PER': 3205, 'I-PER': 2084, 'B-ORG': 0, 'I-ORG': 0, 'B-LOC': 3994, 'I-LOC': 720, 'B-MISC': 2021, 'I-MISC': 648},
    #          {'O': 9478, 'B-PER': 0, 'I-PER': 0, 'B-ORG': 3419, 'I-ORG': 1593, 'B-LOC': 3014, 'I-LOC': 610, 'B-MISC': 1780, 'I-MISC': 552}]

    
    # titles = ['MISC', 'LOC', 'ORG', 'PER']
    # for i in range(1, 5):
    #     stat = stats[i - 1]
        
    #     #plt.subplot(1, 1, i)
    #     plt.title("Class Distribution w/o {}".format(titles[i - 1]))
    #     plt.xticks(range(len(all_tags)), all_tags)
    #     plt.xlabel('Entity Tag')
    #     plt.ylabel('Num. Sentences')
        
    #     plt.bar(range(len(all_tags)), [stat[tag] for tag in all_tags]) 
    #     plt.show()
    # #plt.show()

    stats_one = {'O': 7116, 'B-PER': 737, 'I-PER': 78, 'B-ORG': 1448, 'I-ORG': 44, 'B-LOC': 1618, 'I-LOC': 7, 'B-MISC': 676, 'I-MISC': 90}
    plt.title("Class Distribution, One Tag Only")
    plt.xticks(range(len(all_tags)), all_tags)
    plt.xlabel('Entity Tag')
    plt.ylabel('Num. Sentences')
        
    plt.bar(range(len(all_tags)), [stats_one[tag] for tag in all_tags]) 
    plt.show()