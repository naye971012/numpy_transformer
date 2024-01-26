from typing import Tuple,Any
import numpy as np

from numpy_functions import *
from numpy_models.transformer.feedforward import FeedForward_np

class transformer_decoder_block_np:
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
        """
        transformer decoder has...
        1. masked self attention
        2. encoder decoder attention
        3. feedforward
        4. add&norm
        """
        self.self_attention_layer = Multihead_Attention_np(query_embed_dim=self.embedding_dim,
                                                 key_embed_dim=self.embedding_dim,
                                                 value_embed_dim=self.embedding_dim,
                                                 attention_embed_dim=self.embedding_dim,
                                                 num_heads=self.num_heads,
                                                 is_mask=True, #decoder use masked self attention
                                                 scale=True)
        self.layernorm_layer1 = Layer_Normalization_np()
        
        
        self.encoder_decoder_attention = Multihead_Attention_np(query_embed_dim=self.embedding_dim,
                                                 key_embed_dim=self.embedding_dim,
                                                 value_embed_dim=self.embedding_dim,
                                                 attention_embed_dim=self.embedding_dim,
                                                 num_heads=self.num_heads,
                                                 is_mask=False,
                                                 scale=True)
        self.layernorm_layer2 = Layer_Normalization_np()
        
        
        self.feedforward_layer = FeedForward_np(self.embedding_dim,
                                                exposion_size=4,
                                                drop=True,
                                                drop_p=0.2)

        self.layernorm_layer3 = Layer_Normalization_np()
        
    def forward(self,x:np.array, encoder_output:np.array) -> np.array:
        """
        input and output is same
        
        Args:
            x (np.array): [# of batch, sentence length, embedding_dim]
            encoder_output (np.array): [# of batch, sentence length, embedding_dim]

        Returns:
            np.array: [# of batch, sentence length, embedding_dim]
        """
        
        ############################# EDIT HERE #############################
        pass
    
        #####################################################################
        
        return x
    
    def backward(self, d_prev:np.array) -> Tuple[np.array,np.array]:
        """
        return two d_prev, one is backward value of decoder(d_prev)
        and the other is backward value of encoder(encoder_d_prev)
        
        Args:
            d_prev (np.array): [# of batch, sentence length, embedding_dim]

        Returns:
            d_prev (np.array): [# of batch, sentence length, embedding_dim]
            encoder_d_prev (np.array): [# of batch, sentence length, embedding_dim]
        """
        
        ############################# EDIT HERE #############################
        pass
    
        #####################################################################
        
        #return d_prev, encoder_d_prev
    
    def __call__(self, x:np.array, encoder_output:np.array) -> Tuple[np.array,np.array]:
        return self.forward(x, encoder_output)

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