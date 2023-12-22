from typing import Any
import numpy as np

from numpy_functions import *
from feedforward import FeedForward_np

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
        
        self.self_attention_layer = Multihead_Attention_np(query_embed_dim=self.embedding_dim,
                                                 key_embed_dim=self.embedding_dim,
                                                 value_embed_dim=self.embedding_dim,
                                                 attention_embed_dim=self.embedding_dim,
                                                 num_heads=self.num_heads,
                                                 is_mask=False,
                                                 scale=True)
        self.layernorm_layer1 = Layer_Normalization_np()
        
        self.feedforward_layer = FeedForward_np(self.embedding_dim,
                                                exposion_size=4,
                                                drop=True,
                                                drop_p=0.2)

        self.layernorm_layer2 = Layer_Normalization_np()
        
    def forward(self,x:np.array) -> np.array:
        """
        input and output is same
        
        Args:
            x (np.array): [# of batch, sentence length, embedding_dim]

        Returns:
            np.array: [# of batch, sentence length, embedding_dim]
        """
        
        #attention
        out, att_map = self.self_attention_layer(x, x, x)
        #add & norm
        x = x + out
        x = self.layernorm_layer1(x)
        
        #feedforward
        out = self.feedforward_layer(x)
        #add & norm
        x = x + out
        x = self.layernorm_layer2(x)
        
        return x
    
    def backward(self, d_prev:np.array) -> np.array:
        """
        input and output is same
        
        Args:
            d_prev (np.array): [# of batch, sentence length, embedding_dim]

        Returns:
            np.array: [# of batch, sentence length, embedding_dim]
        """
        
        d_prev = self.layernorm_layer2.backward(d_prev)
        d_prev = self.feedforward_layer.backward(d_prev)
        d_prev = d_prev + 1
        
        d_prev = self.layernorm_layer1.backward(d_prev)
        d_prev = self.self_attention_layer.backward(d_prev)
        d_prev = d_prev + 1
        
        return d_prev
    
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
        
        optimizer.update_grad('tf_encoder_block_layernorm2'+ name , self.layernorm_layer2, LR)
        optimizer.update_grad('tf_encoder_block_layernorm1'+ name , self.layernorm_layer1, LR)
        optimizer.update_grad('tf_encoder_block_ffd'+ name , self.feedforward_layer, LR)
        optimizer.update_grad('tf_encoder_block_selfatt'+ name , self.self_attention_layer, LR)