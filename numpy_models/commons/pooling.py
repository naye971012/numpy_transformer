import numpy as np

class MaxPooling2D:

    def __init__(self, pooling_shape: tuple, stride: int=None) -> None:
        self.pool_w, self.pool_h = pooling_shape
        self.stride = stride
        
        self.d_zeros = None
    
    def forward(self,x):
        self.d_zeros = np.zeros_like(x)
        
        
        return x
    
    def backward(self,d_prev):
        
        
        return self.d_zeros