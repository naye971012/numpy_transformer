from typing import List
import numpy as np
import re

from .vocab import Vocabulary


class Word_tokenizer_np:
    """
    simple tokenizer using vocab
    """
    def __init__(self):
        self.vocab = Vocabulary()
    
    def __len__(self) -> int:
        return len(self.vocab)
    
    def get_vocab(self) -> Vocabulary:
        return self.vocab
    
    def tokenize(self, input_list: List[str]) -> List[List[str]]:
        """
        input: [i am apple, i love apple]
        output: [ [i, am, apple] , [i, love, apple] ]

        Args:
            input_list (List[str]): list of strings

        Returns:
            List[List[str]]: list of tokenized token list
        """
        output_list = list()
        for cur_list in input_list:
            cur_list = self.clean_text(cur_list)
            list_tokenized = cur_list.split(' ') #tokenize by blank
            output_list.append(list_tokenized)
        return output_list
    
    def clean_text(self, sentence:str) -> str:
        """
        convert all text into lowercase and remain only alphabat (for vocab hit ratio)
        """
        sentence = sentence.lower()
        sentence = re.sub(r'[^a-z ]', '', sentence)
        return sentence
    
    def convert_one_ids_to_tokens(self, sentence: List[int]) -> str:
        output = ""
        for word in sentence:
            output = output + self.vocab.idx2word[word] + " "
        return output
    
    def convert_one_tokens_to_ids(self, sentence: List[str]) -> List[int]:
        """
        convert one sentence into ids

        Args:
            sentence (List[str]): list of tokens

        Returns:
            List[int]: list of ids
        """
        output = list()
        for word in sentence:
            output.append(self.vocab(word))
        return output
    
    def convert_tokens_to_ids(self, input_list: List[str], padding=True, max_length=20) -> np.array:
        """
        input: [ [i, am, apple] , [i, love, apple] ]
        output: [ [1, 3, 5] , [1, 6, 5] ]

        Args:
            input_list (List[str]): list of tokenized token list

        Returns:
            np.array(List[List[int]]): list of id list
        """
        output_list = list()
        for cur_list in input_list:
            list_ids = self.convert_one_tokens_to_ids(cur_list)
            
            #when padding, make length==max_length padding blank with zero
            if padding:
                if len(list_ids) < max_length:
                    list_ids += [0] * (max_length - len(list_ids))
                else:
                    list_ids = list_ids[:max_length]
                    
            output_list.append(list_ids)
        
        output_list = [np.array(sublist) for sublist in output_list]
        output_list = np.array(output_list)
        return output_list
    
    def train(self, input_list: List[str]) -> None:
        """
        train vocabulary set using exist data
        
        Args:
            input_list (List[str]): list of sentence you want to train
        """
        vocab_set = list()
        list_tokenized = self.tokenize(input_list)
        
        #add all word into list
        for cur_list in list_tokenized:
            vocab_set.extend(cur_list)
        
        #remove duplicated
        vocab_set = list(set(vocab_set))

        #add all word input vocabulary set
        for cur in vocab_set:
            self.vocab.add_word(cur)