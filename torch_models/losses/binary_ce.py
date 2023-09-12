import numpy as np
import torch
import torch.nn as nn


class Binary_Cross_Entropy_th(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce_loss = torch.nn.BCELoss()
    
    def forward(self, pred, target ):
        """
        assert pred is prob, not logit!!!
        ( prob = [0,1] , log = [-inf,+inf] )
        
        Args:
            x (Tensor): [batch, ... ]

        Returns:
            Tensor: Scaler 
        """

        output = self.bce_loss(pred, target)
        return output