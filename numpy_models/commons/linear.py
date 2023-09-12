import numpy as np

class Linear_np:
    def __init__(self, in_feat , out_feat):
        pass
        
        
    def forward(self, x):
        """
        just one layer Linear function
        
        Args:
            x (array): [# of batch, ... , in_features ]

        Returns:
            array, [# of batch, ... , out_features]
        """
        return self.linear(x)