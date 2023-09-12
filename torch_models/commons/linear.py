import torch
import torch.nn as nn
import numpy as np

class Linear_th(nn.Module):
    def __init__(self, in_feat , out_feat):
        super().__init__()
        
        self.linear = nn.Linear(in_features=in_feat , out_features=out_feat)
        
    def forward(self, x):
        """
        just one layer Linear function
        
        Args:
            x (Tensor): [# of batch, ... , in_features ]

        Returns:
            Tensor, [# of batch, ... , out_features]
        """
        return self.linear(x)