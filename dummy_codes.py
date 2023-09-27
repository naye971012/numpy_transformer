import torch
import numpy as np


class softmax:
    def __init__(self) -> None:
        self.output = None
    def forward(self,x):
        self.output = np.exp(x)
        return self.output
    def backward(self,d_prev):
        
        output = self.output * (1-self.output)
        return d_prev * output