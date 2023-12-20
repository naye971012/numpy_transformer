import numpy as np


class Sigmoid_np():
    """
    sigmoid with numpy
    """
    def __init__(self) -> None:
        self.output = None # save output
        self.grad = None # save gradient
        
    def forward(self,x:np.array) -> np.array:
        """
        forward process of sigmoid
        
        Args:
            x (np.array): [batch, ... ]

        Returns:
            np.array: [batch, ... ]
        """
        out = 1 / ( 1 + np.exp(-x) )
        
        self.output = out
        return out
    
    def backward(self, d_prev:int=1) -> np.array:
        """
        backward process of sigmoid
        
        Args:
            d_prev (np.array): [batch, ... ]

        Returns:
            np.array: [batch, ... ]
        """
    
        grad = d_prev * (self.output) * (1 - self.output )
        
        self.grad = grad
        return self.grad
    
    def __call__(self,x):
        return self.forward(x)