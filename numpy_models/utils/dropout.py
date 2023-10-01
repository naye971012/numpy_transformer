import numpy as np

class dropout_np:
    def __init__(self) -> None:
        pass
    
    def forward(self,x):
        pass
    
    def backward(self,d_prev):
        pass
    
    def __call__(self,x):
        return self.forward(x)