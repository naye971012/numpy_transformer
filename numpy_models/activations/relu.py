import numpy as np


class Relu_np():
    def __init__(self) -> None:
        self.output = None # save output
        self.grad = None # save gradient
        
        self.d_zeros = None # save utils which needs when backward
        
    def forward(self,x):
        """
        Args:
            x (np.array): [batch, ... ]

        Returns:
            np.array: [batch, ... ] 
        """
        self.d_zeros = x<0
        x[self.d_zeros]=0
        
        self.output = x #save output
        return x
    
    def backward(self, d_prev):
        """
        backward:
            output -> d_relu -> grad
        
        """
        prev = np.ones_like(d_prev)
        prev[self.d_zeros] = 0
        
        self.grad = prev * d_prev #save grad
        return self.grad
    
    def __call__(self,x):
        return self.forward(x)