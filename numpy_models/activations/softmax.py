from typing import Any
import numpy as np

class softmax_np:
    def __init__(self) -> None:
        self.output = None
        
    def forward(self,x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True)) 
        softmax_output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        
        self.output = softmax_output
        return self.output
    
    def backward(self,d_prev):
        grad = self.output * d_prev
        sum_grad = np.sum(grad, axis=-1, keepdims=True)
        softmax_grad = grad - self.output * sum_grad
        return softmax_grad
    
    def __call__(self,x):
        return self.forward(x)