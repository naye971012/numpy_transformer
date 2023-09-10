import numpy as np
import torch
import torch.nn as nn


class Binary_Cross_Entropy_th(nn.Module):
    def __init__(self,eps=1e-10) -> None:
        super().__init__()
        self.eps = eps
    
    def forward(self, pred, target ):
        """
        assert pred is prob, not logit!!!
        ( prob = [0,1] , log = [-inf,+inf] )
        
        Args:
            x (Tensor): [batch, ... ]

        Returns:
            Tensor: Scaler 
        """

        output = -1 * torch.mean( pred * torch.log(target + self.eps ) + (1-pred) * torch.log( 1-target + self.eps ))
        return output