from typing import Any
import numpy as np

from numpy_functions import *

class FeedForward_np:
    def __init__(self, 
                 layer_dim:int,
                 exposion_size:int=4,
                 drop:bool=True,
                 drop_p:float=0.2) -> None:
        """
        Args:
            layer_dim (int): dim of input layer
            exposion_size (int, optional): multiplication factor. size of hidden layer. Defaults to 4.
            drop (bool, optional): use dropout. Defaults to True.
        """
        self.layer_dim = layer_dim
        self.expose = exposion_size
        self.drop = drop
        self.drop_p = drop_p
    
        self.init_params()
        
    def init_params(self) -> None:
        
        self.input_layer = Linear_np(self.layer_dim, self.layer_dim * self.expose)
        self.output_layer = Linear_np(self.layer_dim * self.expose, self.layer_dim)
        
        self.activation = Relu_np()
        self.drop_layer = Dropout_np(self.drop_p)
    
    def forward(self,x:np.array) -> np.array:
        
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.drop_layer(x) if self.drop else x
        x = self.output_layer(x)
        
        return x
    
    def backward(self, d_prev:np.array) -> np.array:
        
        d_prev = self.output_layer.backward(d_prev)
        d_prev = self.drop_layer.backward(d_prev) if self.drop else d_prev
        d_prev = self.activation.backward(d_prev)
        d_prev = self.input_layer.backward(d_prev)
        
        return d_prev
    
    def __call__(self, x) -> Any:
        return self.forward(x)