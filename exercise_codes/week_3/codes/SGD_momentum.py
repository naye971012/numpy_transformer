import numpy as np

class SGD_momentum_np:
    """
    implementation of optimizer SGD-with-momentum in numpy
    """
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
        """
        #save current velocity
        self.save_velocity(layer_name,layer)
        
        #update gradient
        #################### edit here ####################
        for param_key, grad_key in zip( sorted(layer.params.keys()), sorted(layer.grads.keys()) ):
            pass
        
        ###################################################

    def save_velocity(self,layer_name, layer):
        """
        save velocity of curren layer
        """
        for param_key, grad_key in zip( sorted(layer.params.keys()), sorted(layer.grads.keys()) ):
              
            #################### edit here ####################
            pass
            
            
            ###################################################
    def step(self):
        pass