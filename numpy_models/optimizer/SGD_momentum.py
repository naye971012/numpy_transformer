import numpy as np

class SGD_momentum_np:
    def __init__(self, alpha: float = 0.9) -> None:
        self.velocity = dict()
        self.alpha = alpha
        """
        alpha: default 0.9
        
        W(t+1) = W(t) + lr * V(t)
        
        V(t) = a * V(t-1) - cost
        
        """
    def update_grad(self, layer_name:str, layer, LR:float, have_db:bool):
        """

        Args:
            layer_name (str): _description_
            layer (_type_): layer(ex.)
            LR (float): Learning rate
            have_db (bool): layer에 dW외에 db가 있는지 유무, default=True
        """
        self.save_velocity(layer_name,layer,have_db)
        
        layer.W = layer.W + LR * self.velocity[f"{layer_name}_dW"]
        
        if not have_db:
            return layer
        
        layer.b = layer.b + LR * self.velocity[f"{layer_name}_db"]
        
        return layer

    def save_velocity(self,layer_name, layer, have_db):
        
        if layer_name not in self.velocity.keys():
            self.velocity[f"{layer_name}_dW"] = -1 * layer.dW
        else:
            self.velocity[f"{layer_name}_dW"] = self.alpha * self.velocity[f"{layer_name}_dW"] - layer.dW

        
        if not have_db:
            return
        
        if layer_name not in self.velocity.keys():
            self.velocity[f"{layer_name}_db"] = -1 * layer.db
        else:
            self.velocity[f"{layer_name}_db"] = self.alpha * self.velocity[f"{layer_name}_db"] - layer.db