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
    def update_grad(self, layer_name:str, layer, LR:float):
        """

        Args:
            layer_name (str): _description_
            layer (_type_): layer(ex.)
            LR (float): Learning rate
            have_db (bool): layer에 dW외에 db가 있는지 유무, default=True
        """
        if not hasattr(layer, 'params'):
            return  # Break the process as there are no parameters to update

        self.save_velocity(layer_name,layer)
        
        for param_key, grad_key in zip( sorted(layer.params.keys()), sorted(layer.grads.keys()) ):
            layer.params[param_key] += LR * self.velocity[layer_name + grad_key]
        
        
    def save_velocity(self,layer_name, layer):
        
        for param_key, grad_key in zip( sorted(layer.params.keys()), sorted(layer.grads.keys()) ):
              
            if (layer_name + grad_key) not in self.velocity.keys():
                self.velocity[layer_name + grad_key] = -1 * layer.grads[grad_key]
            else:
                self.velocity[layer_name + grad_key] = self.alpha * self.velocity[layer_name + grad_key] - layer.grads[grad_key]

    def step(self):
        pass