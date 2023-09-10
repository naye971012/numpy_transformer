import numpy as np
import torch
import torch.nn as nn

class Sigmoid_th(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self,x):
        """
        Args:
            x (Tensor): [batch, ... ]

        Returns:
            Tensor: [batch, ... ] 
        """
        output = 1 / ( 1 + torch.exp(-x) )
        return output