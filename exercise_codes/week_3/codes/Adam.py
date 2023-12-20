import numpy as np

class Adam_np:
    """
    implementation of optimizer Adam in numpy
    """
    def __init__(self, alpha: float = 0.001,
                       beta1: float = 0.99,
                       beta2: float = 0.999,
                       eps: float = 1e-8) -> None:
        self.velocity = dict()
        self.momentum = dict()
        
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        self.t = 1 # step
        
        """
        M(t) = b1 * M(t-1) + (1-b1) * cost
        V(t) = b2 * V(t-1) + (1-b2) * (cost**2)
        
        M_hat(t) = M(t) / (1 - (b1**t))
        V_hat(t) = V(t) / (1 - (b2**t))
        
        W(t+1) = W(t) - LR * M_hat / sqrt(V_hat + eps)
        
        """
    def step(self):
        self.t = self.t + 1
    
    def update_grad(self, layer_name:str, layer, LR:float) -> None:
        
        """
        Args:
            layer_name (str): _description_
            layer (_type_): layer
            LR (float): Learning rate
        """
        #save velocity/momentum
        self.save_velocity(layer_name,layer)
        self.save_momentum(layer_name,layer)

        #update layer gradient
        for param_key, grad_key in zip( sorted(layer.params.keys()), sorted(layer.grads.keys()) ):
            name = (layer_name + grad_key)

            #################### edit here ####################
            
            pass
            ###################################################

    def save_velocity(self,layer_name, layer):
        """
        save velocity of current layer
        """
        for param_key, grad_key in zip( sorted(layer.params.keys()), sorted(layer.grads.keys()) ):
            name = (layer_name + grad_key)
            
            #################### edit here ####################
            
            pass
            ###################################################

    
    def save_momentum(self,layer_name, layer):
        """
        save momentum of current layer
        """
        for param_key, grad_key in zip( sorted(layer.params.keys()), sorted(layer.grads.keys()) ):
            name = (layer_name + grad_key)

            #################### edit here ####################
            
            pass
            ###################################################
                        