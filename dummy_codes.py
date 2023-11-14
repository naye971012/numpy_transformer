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

def normalize(x):                              # Normalization
    
    mean = x.mean(axis = 1).reshape(-1,1)
    std = x.std(axis = 1).reshape(-1,1)
    
    return (x - mean) / (std + 10**-100)

def backward_normalize(x, grad_output):
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    
    dx = (1.0 / (std + 1e-100)) * grad_output
    dmean = -dx.sum(axis=1, keepdims=True)
    dstd = -np.sum(((x - mean) / (std + 1e-100)) * dx, axis=1, keepdims=True) / (std + 1e-100)
    
    dx += (1.0 / x.shape[1]) * (x - mean) * dstd
    
    return dx, dmean, dstd

# 사용 예시
x = np.array([[1, 2], [3, 4], [5, 6]])
grad_output = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

normalized_x = normalize(x)
dx, dmean, dstd = backward_normalize(x, grad_output)

print("Original x:\n", x)
print("Normalized x:\n", normalized_x)
print("Gradient w.r.t. x:\n", dx)
print("Gradient w.r.t. mean:\n", dmean)
print("Gradient w.r.t. std:\n", dstd)