import numpy as np

class Numpymodel_np:
    def __init__(self) -> None:
        pass
    def forward(self,x:np.array) -> np.array:
        pass
    def backward(self,d_prev:np.array) -> np.array:
        pass
    def __call__(self,x):
        return self.forward(x)