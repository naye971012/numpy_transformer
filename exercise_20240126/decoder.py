from typing import Any, Tuple
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
from numpy_models.transformer.decoder_block import transformer_decoder_block_np


class transformer_decoder_np:
    def __init__(self,
                 num_blocks:int = 4,
                 sentence_length:int=50,
                 embedding_dim:int=128,
                 num_heads:int=4,
                 vocab_size:int=10000) -> None:
        """
        
        Args:
            num_blocks (int, optional): num of block in encoder. Defaults to 4.
            sentence_length (int, optional): length of sentence . Defaults to 50.
            embedding_dim (int, optional): embedding dimension. Defaults to 128.
            num_heads (int, optional): num of head in attention. Defaults to 4.
            vocab_size (int, optional): num of vocab size in tokenizer. Defaults to 10000.
        """
        assert embedding_dim%num_heads==0, "embedding dim should be divisiable to num_heads"
        
        self.num_blocks = num_blocks
        
        self.sentence_length = sentence_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.vocab_size = vocab_size
    
        self.init_params()
    
    def init_params(self):
        
        ############################# EDIT HERE #############################
        pass
    
        #####################################################################
        
    def forward(self,x:np.array, encoder_x:np.array) -> np.array:
        """
        Args:
            x (np.array): [# of batch, sentence length]
            encoder_x (np.array): [# of batch, sentence length, embedding dim]
        
        Returns:
            np.array: [# of batch, sentence length, vocab size]
        """
        ############################# EDIT HERE #############################
        pass
    
        #####################################################################
        return x
    
    def backward(self, d_prev:np.array) -> Tuple[None,np.array]:
        """
        Args:
            d_prev (np.array): [# of batch, sentence length, vocab size]

        Returns:
            None
            encoder_d_prev (np.array): [# of batch, sentence length, embedding dim]
        """
        ############################# EDIT HERE #############################
        pass
    
        #####################################################################
    
        #return d_prev, encoder_d_prev_avg
    
    def __call__(self, x:np.array, encoder_out:np.array) -> np.array:
        return self.forward(x, encoder_out)
    
    def update_grad(self,
                    name:str="1",
                    optimizer:Any=SGD_np,
                    LR:float=0.001):
        """
        update weight recursively
        
        if layer is in numpy_functions, update gradient
        else(layer is a block), call update_grad function recursively
        
        Args:
            name (str): distinguish value
            optimizer (Any): your optimizer
            lr (float): learning rate
        """
        
        ############################# EDIT HERE #############################
        pass
    
        #####################################################################


if __name__=="__main__":
    encoder = transformer_decoder_np(num_blocks=4,
                                     sentence_length=50,
                                     embedding_dim=128,
                                     num_heads=4,
                                     vocab_size=10000)
    
    x = np.random.randint(0,100,size=(10,50))
    
    sgd = SGD_np()
    
    out = encoder(x, encoder.embedding_layer(x))
    encoder.backward(out)
    
    encoder.update_grad("1",sgd,0.001)