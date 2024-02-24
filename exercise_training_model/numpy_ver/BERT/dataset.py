from typing import Any, Tuple, List
import sys
import os
import argparse
import numpy as np

############## this block is just for import moudles ######
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
grand_path = os.path.dirname(parent_path)
sys.path.append(grand_path)
###########################################################

from numpy_functions import *


class CustomDataset:
    def __init__(self,
                 args,
                 data:List, 
                 tokenizer:Word_tokenizer_np) -> None:
        
        self.args = args
        self.data = data
        self.tokenizer = tokenizer
        self.vocab = self.tokenizer.get_vocab()
        
    def __getitem__(self,idx:int):
        """
        return (input, output) tuple
        
        input/output: List[int] with token id
        """
        # type(sentence) = List[str]
        sentence = self.tokenizer.tokenize( [self.data[idx]] )[0] #function input type is list
        
        # type(_output) = List[int]
        _output = self.tokenizer.convert_tokens_to_ids([ ['[CLS]'] + sentence + ['[SEP]'] ], 
                                                      padding=True, 
                                                      max_length=self.args.max_len)[0]
        
        # type(_input) = List[str]
        _input = self.process_caption(sentence)
        
        # type(_input) = List[int]
        _input = self.tokenizer.convert_tokens_to_ids([_input], 
                                                      padding=True, 
                                                      max_length=self.args.max_len)[0]
        
        return _input, _output
    
    def __len__(self):
        return len(self.data)

    def process_caption(self, sentence_token:List[str]):
        """
        args:   
            sentence_token: [i, am, sentence]      
        return:
            list of str with augmented: [Mask, am, sentence]
        """
        output_tokens = []
        deleted_idx = []

        for i, token in enumerate(sentence_token):
            prob = np.random.random()

            if prob < 0.20:
                prob /= 0.20

                # 80% randomly change token to mask token
                if prob < 0.9:
                    output_tokens.append("[MASK]")
                # 10% randomly change token to random token
                else: #prob < 0.9:
                    output_tokens.append(np.random.choice( list( self.vocab.word2idx.keys() ) ) )
                    # -> rest 10% randomly keep current token
                #else:
                #    output_tokens.append(token)
                #    deleted_idx.append(len(output_tokens) - 1)
            else:
                # no masking token (will be ignored by loss function later)
                output_tokens.append(token)

        if len(deleted_idx) != 0:
            output_tokens = [output_tokens[i] for i in range(len(output_tokens)) if i not in deleted_idx]

        output_tokens = ['[CLS]'] + output_tokens + ['[SEP]']

        return output_tokens


class CustomDataloader:
    """
        my custom dataloader class
        if len(dataset)%batch_size !=0, ignore remain items
    """
    def __init__(self, dataset, batch_size:int, shuffle:bool) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __getitem__(self, idx: int):
        output = {
            "input" : [],
            "label" : []
        }
        for i in range(self.batch_size * idx , self.batch_size * (idx+1), 1 ):
            _input, _output = self.dataset[i]
            output["input"].append(_input)
            output["label"].append(_output)

        return output
    
    def __len__(self) ->int:
        return len(self.dataset) // self.batch_size
