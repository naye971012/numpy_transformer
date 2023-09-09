import numpy as np
import torch
import torch.nn as nn

class Relu_th(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self,x):
        """
        Args:
            x (Tensor): [batch, ... ]

        Returns:
            Tensor: [batch, ... ] 
        """
        x = torch.where(x<0, 0, x)
        return x