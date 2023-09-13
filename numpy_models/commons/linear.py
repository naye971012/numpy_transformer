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
        return
    
    def backward(self):
        """
        backward:
            input <-- linear <-- output
        
        """
        return
    
    def __call__(self, pred, target):
        return self.forward(pred, target)