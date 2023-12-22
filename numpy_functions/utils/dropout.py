import numpy as np

class Dropout_np:
    """
    dropout layer implemented with numpy
    """
    
    def __init__(self, prob:float) -> None:
        #dropout percent
        assert (prob>=0 and prob<1)
        self.drop_prob = prob
        
        self.drop_mask = None
    
    def forward(self,x:np.array) -> np.array:
        """
        forward process of dropout layer
        
        Args:
            x (np.array): [# of batch, ... ]

        Returns:
            np.array: [# of batch, ... ]
            
        """
        self.drop_mask = np.random.rand(*(x.shape)) > self.drop_prob
        x = x * self.drop_mask
        x = x / self.drop_prob
        
        return x
    
    def backward(self,d_prev:np.array) -> np.array:
        """
        backward process of dropout layer
        
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