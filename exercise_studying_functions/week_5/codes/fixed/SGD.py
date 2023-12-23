import numpy as np

class SGD_np:
    def __init__(self) -> None:
        """
        
        W(t+1) = W(t) - lr * cost
        
        """
        
    def update_grad(self, layer_name:str, layer, LR:float):
        """

        Args:
            layer_name (str): _description_
            layer (_type_): layer(ex.)
            LR (float): Learning rate
        """
        
        for param_key, grad_key in zip( sorted(layer.params.keys()), sorted(layer.grads.keys()) ):
            layer.params[param_key] -= LR * layer.grads[grad_key]
        
        
    def step(self):
        pass