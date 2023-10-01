import numpy as np

class Embedding_np:
    def __init__(self) -> None:
        pass
    def forward(self,x) -> np.array:
        pass
    def backward(self,d_prev) -> np.array:
        pass
    def __call__(self,x):
        return self.forward(x)