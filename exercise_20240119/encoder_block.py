from typing import Any
import numpy as np

from numpy_functions import *
from numpy_models.transformer.feedforward import FeedForward_np

class transformer_encoder_block_np:
    def __init__(self, 
                 embedding_dim:int=128,
                 num_heads:int=4
                 ) -> None:
        """
        Args:
            sentence_length (int): _description_. Defaults to 50.
            vocab (Vocabulary): _description_. Defaults to None.
        """
        assert embedding_dim%num_heads==0, "embedding dim should be divisiable to num_heads"
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
    
        self.init_params()
    
    def init_params(self):
        
        ################### EDIT HERE ##########################
        
        pass
        ########################################################
        
    def forward(self,x:np.array) -> np.array:
        """
        input and output is same
        
        Args:
            x (np.array): [# of batch, sentence length, embedding_dim]

        Returns:
            np.array: [# of batch, sentence length, embedding_dim]
        """
        
        ################### EDIT HERE ##########################
        
        pass
        ########################################################
    
    def backward(self, d_prev:np.array) -> np.array:
        """
        input and output is same
        
        Args:
            d_prev (np.array): [# of batch, sentence length, embedding_dim]

        Returns:
            np.array: [# of batch, sentence length, embedding_dim]
        """
        
        ################### EDIT HERE ##########################
        
        pass
        ########################################################
    
    def __call__(self, x:np.array) -> np.array:
        return self.forward(x)

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
        
        ################### EDIT HERE ##########################
        
        pass
        ########################################################