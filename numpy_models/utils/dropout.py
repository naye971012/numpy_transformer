import numpy as np

class Dropout_np:
    def __init__(self, prob:float) -> None:
        assert (prob>=0 and prob<1)
        
        self.drop_prob = prob
        self.drop_mask = None
    
    def forward(self,x:np.array) -> np.array:
        """

        Args:
            x (np.array): [# of batch, ... ]

        Returns:
            np.array: [# of batch, ... ]
            
        """
        self.drop_mask = np.random.rand(*x) > self.drop_prob
        x = x * self.drop_mask
        x = x / self.drop_prob
        
        return x
    
    def backward(self,d_prev:np.array) -> np.array:
        """

        Args:
            d_prev (np.array): [# of batch, ... ]

        Returns:
            np.array: [# of batch, ... ]
            
        """
        d_prev = d_prev * self.drop_mask
        d_prev = d_prev / self.drop_prob
        
        return d_prev
    
    def __call__(self,x):
        return self.forward(x)