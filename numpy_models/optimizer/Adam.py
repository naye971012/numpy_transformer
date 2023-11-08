import numpy as np

class Adam_np:
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
    
    def update_grad(self, layer_name:str, layer, LR:float, have_db:bool):
        
        """

        Args:
            layer_name (str): _description_
            layer (_type_): layer(ex.)
            LR (float): Learning rate
            have_db (bool): layer에 dW외에 db가 있는지 유무, default=True
        """
        self.save_velocity(layer_name,layer,have_db)
        self.save_momentum(layer_name,layer,have_db)
        
        momentum_hat = self.momentum[f"{layer_name}_dW"] / (1 - (self.beta1 ** self.t) )
        velocity_hat = self.velocity[f"{layer_name}_dW"] / (1 - (self.beta2 ** self.t) )
        layer.W = layer.W - LR * momentum_hat / (np.sqrt(velocity_hat + self.eps) )
        
        if not have_db:
            return layer
        
        momentum_hat = self.momentum[f"{layer_name}_db"] / (1 - (self.beta1 ** self.t) )
        velocity_hat = self.velocity[f"{layer_name}_db"] / (1 - (self.beta2 ** self.t) )
        layer.b = layer.b - LR * momentum_hat / (np.sqrt(velocity_hat + self.eps) )
        
        return layer

    def save_velocity(self,layer_name, layer, have_db):
        
        if layer_name not in self.velocity.keys():
            self.velocity[f"{layer_name}_dW"] = (1-self.beta2) * (layer.dW **2)
        else:
            self.velocity[f"{layer_name}_dW"] = self.beta2 * self.velocity[f"{layer_name}_dW"] + \
                                                (1-self.beta2) * (layer.dW **2)

        
        if not have_db:
            return
        
        if layer_name not in self.velocity.keys():
            self.velocity[f"{layer_name}_db"] = (1-self.beta2) * (layer.db **2)
        else:
            self.velocity[f"{layer_name}_db"] = self.beta2 * self.velocity[f"{layer_name}_db"] + \
                                                (1-self.beta2) * (layer.db **2)
    
    def save_momentum(self,layer_name, layer, have_db):
        
        if layer_name not in self.momentum.keys():
            self.momentum[f"{layer_name}_dW"] = (1-self.beta1) * (layer.dW)
        else:
            self.momentum[f"{layer_name}_dW"] = self.beta1 * self.momentum[f"{layer_name}_dW"] + \
                                                (1-self.beta1) * (layer.dW)

        
        if not have_db:
            return
        
        if layer_name not in self.momentum.keys():
            self.momentum[f"{layer_name}_db"] = (1-self.beta1) * (layer.db)
        else:
            self.momentum[f"{layer_name}_db"] = self.beta1 * self.momentum[f"{layer_name}_db"] + \
                                                (1-self.beta1) * (layer.db)