from typing import List

class Vocabulary:
    """simple vocab set"""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.init_special_token()
        
    def init_special_token(self):
        self.add_word('<pad>')
        self.add_word('<unk>')
        self.add_word('<cls>')
        self.add_word('<eos>')
        
    def add_word(self, word: str):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word: str) -> int:
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)