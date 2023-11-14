"""
change algorithm of adam for code visability    

"""

import numpy as np

class Adam_new_np:
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
    
    def update_grad(self, layer_name:str, layer, LR:float):
        
        """
        Args:
            layer_name (str): _description_
            layer (_type_): layer(ex.)
            LR (float): Learning rate
        """
        self.save_velocity(layer_name,layer)
        self.save_momentum(layer_name,layer)

        for i, param in enumerate(layer.params):
            name = f"{layer_name}_{i}"

            momentum_hat = self.momentum[name] / (1 - (self.beta1 ** self.t) )
            velocity_hat = self.velocity[name] / (1 - (self.beta2 ** self.t) )
            param = param - LR * momentum_hat / (np.sqrt(velocity_hat + self.eps) )

            layer.params[i] = param # update value
        
        return layer

    def save_velocity(self,layer_name, layer):
        
        for i, grad in enumerate(layer.grads):
            name = f"{layer_name}_{i}"
            
            if name not in self.velocity.keys():
                self.velocity[name] = (1-self.beta2) * (grad **2)
            else:
                self.velocity[name] = self.beta2 * self.velocity[name] + \
                                                    (1-self.beta2) * (grad **2)

    
    def save_momentum(self,layer_name, layer):

        for i, grad in enumerate(layer.grads):
            name = f"{layer_name}_{i}"

            if name not in self.momentum.keys():
                self.momentum[name] = (1-self.beta1) * (grad)
            else:
                self.momentum[name] = self.beta1 * self.momentum[name] + \
                                                    (1-self.beta1) * (grad)
                        