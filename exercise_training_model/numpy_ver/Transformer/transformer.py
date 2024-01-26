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
from numpy_models.transformer import *

class transformer_np:
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
        
        self.encoder = transformer_encoder_np(num_blocks=self.num_blocks,
                                              sentence_length=self.sentence_length,
                                              embedding_dim=self.embedding_dim,
                                              num_heads=self.num_heads,
                                              vocab_size=self.vocab_size)
        
        self.decoder = transformer_decoder_np(num_blocks=self.num_blocks,
                                              sentence_length=self.sentence_length,
                                              embedding_dim=self.embedding_dim,
                                              num_heads=self.num_heads,
                                              vocab_size=self.vocab_size)


    def forward(self,encoder_x:np.array, decoder_x:np.array) -> np.array:
        """
        Args:
            encoder_x (np.array): encoder input x[# of batch, sentence length]
            decoder_x (np.array): decoder input x[# of batch, sentence length]

        Returns:
            prediction (np.array): [# of batch, sentence length, vocab size]
        """
        
        encoder_out = self.encoder(encoder_x)
        
        decoder_out = self.decoder(decoder_x, encoder_out)
        
        return decoder_out
        
    
    def backward(self, d_prev:np.array) -> None:
        """
        Args:
            d_prev (np.array): [# of batch, sentence length, vocab size]

        Returns:
            None
        """        
        
        _, encoder_d_prev = self.decoder.backward(d_prev)
        
        d_prev = self.encoder.backward(encoder_d_prev)
        
        return d_prev
    
    def __call__(self, encoder_x:np.array, decoder_x:np.array) -> np.array:
        """
        Args:
            encoder_x (np.array): encoder input x[# of batch, sentence length]
            decoder_x (np.array): decoder input x[# of batch, sentence length]

        Returns:
            prediction (np.array): [# of batch, sentence length, vocab size]
        """
        return self.forward(encoder_x, decoder_x)
    
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
        
        self.encoder.update_grad(name , optimizer, LR)
        self.decoder.update_grad(name , optimizer, LR)


#for check dimension
if __name__=="__main__":
    transformer = transformer_np(num_blocks=4,
                                     sentence_length=50,
                                     embedding_dim=128,
                                     num_heads=4,
                                     vocab_size=10000)
    
    x = np.random.randint(0,100,size=(10,50))
    
    sgd = SGD_np()
    
    out = transformer(x,x)
    
    transformer.backward(out)
    transformer.update_grad("1",sgd,0.001)