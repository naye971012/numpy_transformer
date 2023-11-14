import numpy as np

class SGD_np:
    def __init__(self) -> None:
        """
        
        W(t+1) = W(t) - lr * cost
        
        """
        
    def update_grad(self, layer_name:str, layer, LR:float, have_db:bool):
        """

        Args:
            layer_name (str): _description_
            layer (_type_): layer(ex.)
            LR (float): Learning rate
            have_db (bool): layer에 dW외에 db가 있는지 유무, default=True
        """
        
        layer.W = layer.W - LR * layer.dW
        
        if not have_db:
            return layer
        
        layer.b = layer.b - LR * layer.db
        
        return layer
    
    def step(self):
        pass