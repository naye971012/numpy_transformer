from typing import List

class Vocabulary:
    """simple vocab set"""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.init_special_token()
        
    def init_special_token(self):
        """
        basic special tokens
        """
        self.add_word('[MASK]') #mask token
        self.add_word('[SEP]') #end of data
        self.add_word('[PAD]') #padding token
        self.add_word('[UNK]') #unknworn token
        self.add_word('[CLS]') #class (first) token
        self.add_word('[EOS]') #end of sentence
        
    def add_word(self, word: str):
        """
        add word in vocabulary set

        Args:
            word (str): word you want to add
        """
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word: str) -> int:
        """
        return vocab index of the word

        Args:
            word (str): word you select

        Returns:
            int: vocab index of word
        """
        if word not in self.word2idx:
            return self.word2idx['[UNK]']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)