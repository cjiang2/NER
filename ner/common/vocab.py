"""
Vocabulary class.

"""

from collections import Counter

def process_text(text: str):
    """Processes a caption string into a list of tonenized words.
    Args:
        caption: A string caption.
    Returns:
        A list of strings; the tokenized caption.
    """
    if isinstance(text, str):
        import nltk
        tokenized_text = []
        tokenized_text.extend(nltk.tokenize.word_tokenize(text.lower()))
    else:
        tokenized_text = text

    return tokenized_text

def maxlen(texts):
    """Return maximum sequence length.
    """
    return max([len(process_text(text)) for text in texts])

class Vocabulary(object):
    """
    Always reserve for pad and start token.
    start_word: str = "<s>",
    pad_word: str = "<pad>",
    unk_word: str = "<unk>",
    """
    def __init__(self,
                 pad_word: str = "<pad>",   # Also used as end token
                ):
        self.pad_word = pad_word
        assert pad_word is not None

        self.vocab = dict()
        self.reverse_vocab = []

        # Save special word ids
        self._save_special_word_ids()

    def _save_special_word_ids(self):
        self.pad_id = 0
        # Pad special tokens
        self.reverse_vocab.append(self.pad_word)
        self.vocab[self.pad_id] = self.pad_word


    def construct(
        self, 
        texts: list,
        min_word_count: int = 0,
        ):
        """Construct vocabulary from a list of texts.
        """
        print("Creating vocabulary...")

        counter = Counter()
        for text in texts:
            tokenized_text = process_text(text)
            counter.update(tokenized_text)
        print("Total words:", len(counter))

        # Filter uncommon words and sort by descending count.
        word_counts = [x for x in counter.items() if x[1] >= min_word_count]
        #print(word_counts)
        word_counts.sort(key=lambda x: x[1], reverse=True)
        print("Words in vocabulary:", len(word_counts))

        # Create the vocabulary dictionary
        for i, x in enumerate(word_counts):
            self.reverse_vocab.append(x[0])
            self.vocab[x[0]] = i + 1
        print("Done.")

    def word_to_id(self, 
                   word: str):
        """Return integer index for a word string
        in vocabulary.
        """
        if word in self.vocab:
            return self.vocab[word]
        else:
            return None

    def id_to_word(self, 
                   word_id: int):
        """Return word string associated w/ the integer word id.
        """
        if word_id >= len(self.vocab) or word_id < 0:
            return None
        else:
            return self.reverse_vocab[word_id]

    def text_to_id(
        self, 
        text: str,
        ):
        """Return integer indices for a text of words.
        """
        tokenized_text = process_text(text)
        return [self.word_to_id(token) for token in tokenized_text]

    def __len__(self):
        return len(self.reverse_vocab)

    def __str__(self):
        return str(self.reverse_vocab[:50]) + " " + str([i for i in range(len(self.reverse_vocab[:50]))])