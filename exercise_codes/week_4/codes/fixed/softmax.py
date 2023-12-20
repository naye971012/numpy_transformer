from typing import Any
import numpy as np

class softmax_np:
    """
    activation softmax with numpy
    """
    def __init__(self) -> None:
        self.output = None
        
    def forward(self,x:np.array) -> np.array:
        """
        forward process of softmax
        
        Args:
            x (np.array): [batch, ... ]

        Returns:
            np.array: [batch, ... ]
        """
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True)) 
        softmax_output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        
        self.output = softmax_output
        return self.output
    
    def backward(self,d_prev:int=1) -> np.array:
        """
        backward process of softmax
        
        Args:
            d_prev (np.array): [batch, ... ]

        Returns:
            np.array: [batch, ... ]
        """
        grad = self.output * d_prev
        sum_grad = np.sum(grad, axis=-1, keepdims=True)
        softmax_grad = grad - self.output * sum_grad
        return softmax_grad
    
    def __call__(self,x):
        return self.forward(x)