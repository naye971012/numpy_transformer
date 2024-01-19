import sys
import os
from typing import Any
import numpy as np

############## this block is just for import moudles ######
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
grand_path = os.path.dirname(parent_path)
sys.path.append(grand_path)
###########################################################

from numpy_functions import *
from numpy_models.transformer import transformer_encoder_np


class Bert_np:
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
        
        self.init_param()
    
    def init_param(self):
        """
        use transformer encoder and cross entropy for MLM task
        """
        self.encoder = transformer_encoder_np(num_blocks=self.num_blocks,
                                              sentence_length=self.sentence_length,
                                              embedding_dim=self.embedding_dim,
                                              num_heads=self.num_heads,
                                              vocab_size=self.vocab_size)
        self.criterion = Cross_Entropy_np()

        self.softmax = softmax_np()
        self.output_layer = Linear_np(self.embedding_dim, self.vocab_size)
        
        
    def forward(self, x:np.array) -> np.array:
        """
        Args:
            x (np.array): [# of batch, sentence length]

        Returns:
            np.array: [# of batch, sentence length, vocab size]
        """
        x = self.encoder.forward(x)
        
        x = self.output_layer(x)
        x = self.softmax(x)
        
        return x
    
    def predict(self, x:np.array) -> np.array:
        """
        Args:
            x (np.array): [# of batch, sentence length]

        Returns:
            np.array: [# of batch, sentence length]
        """
        #x = self.encoder(x)
        x = np.argmax(x, axis=2)
        return x
        
    def loss(self,
             pred:np.array,
             label:np.array) -> float:
        """
        use cross entropy for loss

        Args:
            pred (np.array): [# of batch, sentence length, vocab size]
            label (np.array): [# of batch, sentence length]

        Returns:
            float: average loss
        """
        loss = self.criterion(pred,label)
        return loss
    
    def backward(self, d_prev=1) -> None:
        """
        backward process of bert model
        """
        d_prev = self.criterion.backward(1)
        
        d_prev = self.softmax.backward(d_prev)
        d_prev = self.output_layer.backward(d_prev)
        
        d_prev = self.encoder.backward(d_prev)
        return d_prev
    
    def __call__(self, x:np.array) -> np.array:
        return self.forward(x)
    
    def update_grad(self, 
                    optimizer:Any, 
                    LR:float):
        """
        update weight recursively
        
        if layer is in numpy_functions, update gradient
        else(layer is a block), call update_grad function recursively
        
        Args:
            name (str): distinguish value
            optimizer (Any): your optimizer
            lr (float): learning rate
        """
        optimizer.update_grad('tf_Bert_softmax' , self.softmax, LR)
        optimizer.update_grad('tf_Bert_outputlayer' , self.output_layer, LR)
        
        self.encoder.update_grad(name="Bert",
                                 optimizer=optimizer,
                                 LR=LR)