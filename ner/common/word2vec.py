import numpy as np

def get_embed_matrix(fpath, 
                     embed_dim,
                     vocab,
                     special_words={}):
    """Construct embedding weights from provided vocabulary and word2vec.
    """
    from gensim.models import KeyedVectors
    def load_word2vec(fpath):
        """Load embeddings weights in word2vec format.
        """     
        return KeyedVectors.load_word2vec_format(fpath, binary=True)

    # Terminate loading process if not expected
    if embed_dim != 300 or fpath is None:
        return None

    # Load word2vec
    print('Loading word2vec...')
    embed_index = load_word2vec(fpath)
    print('Loaded.')

    failed = []
    # Prepare embed_matrix
    embed_matrix = np.random.uniform(-0.1, 0.1, (len(vocab), embed_dim))
    for word, i in vocab.vocab.items():
        if word in special_words:
            word_old = word
            word = special_words[word]
            #print('Updated {} using {}, '.format(word_old, word), end='')
        #print('Adding {}, idx {}'.format(word, i))
        try:
            embed_vec = embed_index[word]
            embed_matrix[i] = embed_vec
        except: 
            #print('!Failed to add {}, idx {}'.format(word, i))
            failed.append(word)
            continue
    print('embed_matrix constructed.')
    print('words not found:', len(failed))
    return embed_matrix