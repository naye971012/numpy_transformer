from typing import Any
import sys
import os
import numpy as np

############## this block is just for import moudles ######
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
grand_path = os.path.dirname(parent_path)
sys.path.append(grand_path)
###########################################################

from numpy_functions import *
from encoder_block import transformer_encoder_block_np


class transformer_encoder_np:
    def __init__(self,
                 num_blocks:int = 4,
                 sentence_length:int=50,
                 embedding_dim:int=128,
                 num_heads:int=4,
                 vocab_size:int=10000) -> None:
        """
        Args:
            sentence_length (int): _description_. Defaults to 50.
            vocab (Vocabulary): _description_. Defaults to None.
        """
        assert embedding_dim%num_heads==0, "embedding dim should be divisiable to num_heads"
        
        self.num_blocks = num_blocks
        
        self.sentence_length = sentence_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.vocab_size = vocab_size
    
        self.init_params()
    
    def init_params(self):
        
        self.embedding_layer = Embedding_with_positional_encoding_np(num_emb=self.vocab_size,
                                                                     num_dim=self.embedding_dim)

        self.block_layer = [transformer_encoder_block_np(embedding_dim=self.embedding_dim,
                                                         num_heads=self.num_heads) for _ in range(self.num_blocks)]

        self.softmax = softmax_np()
        self.output_layer = Linear_np(self.embedding_dim, self.vocab_size)
        
    def forward(self,x:np.array) -> np.array:
        """
        Args:
            x (np.array): [# of batch, sentence length]

        Returns:
            np.array: [# of batch, sentence length, vocab size]
        """
        x = self.embedding_layer(x)

        for i in range(self.num_blocks):
            x = self.block_layer[i](x)
        
        x = self.output_layer(x)
        x = self.softmax(x)
        
        return x
    
    def backward(self, d_prev:np.array) -> np.array:
        
        d_prev = self.softmax.backward(d_prev)
        d_prev = self.output_layer.backward(d_prev)
        
        for i in range(self.num_blocks-1,-1,-1):
            d_prev = self.block_layer[i].backward(d_prev)
        d_prev =self.embedding_layer.backward(d_prev)
        
        return d_prev
    
    def __call__(self, x:np.array) -> np.array:
        return self.forward(x)



if __name__=="__main__":
    encoder = transformer_encoder_np(num_blocks=4,
                                     sentence_length=50,
                                     embedding_dim=128,
                                     num_heads=4,
                                     vocab_size=10000)
    
    x = np.random.randint(0,100,size=(10,50))
    
    out = encoder(x)
    encoder.backward(out)