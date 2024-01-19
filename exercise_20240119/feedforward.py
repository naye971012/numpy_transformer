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
        
        ################### EDIT HERE ##########################
        
        pass
        ########################################################
    
    def forward(self,x:np.array) -> np.array:
        
        ################### EDIT HERE ##########################
        
        pass
        ########################################################
        
        return x
    
    def backward(self, d_prev:np.array) -> np.array:
        
        ################### EDIT HERE ##########################
        
        pass
        ########################################################
        
        return d_prev
    
    def __call__(self, x) -> Any:
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