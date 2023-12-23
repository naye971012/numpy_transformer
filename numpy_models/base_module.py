import numpy as np

class module_np:
    def __init__(self) -> None:
        pass
    
    def init_param(self):
        pass
    
    def forward(self, x:np.array) -> np.array:
        pass
    
    def predict(self, x:np.array) -> np.array:
        pass
    
    def loss(self,
             pred:np.array,
             label:np.array) -> np.array:
        pass
    
    def backward(self, d_prev=1) -> None:
        pass
    
    def __call__(self, x:np.array) -> np.array:
        pass
    
    def update_grad(self):
        pass