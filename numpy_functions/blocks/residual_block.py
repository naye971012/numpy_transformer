from typing import Any
import numpy as np

class Residual_block_np:
    def __init__(self, model) -> None:
        self.model = model
    
    def forward(self,x:np.array):
        out = self.model(x)
        return out + x
    
    def backward(self,d_prev):
        d_out = self.model.backward(d_prev)
        return 1 + d_out
    
    def __call__(self,x):
        return self.forward(x)