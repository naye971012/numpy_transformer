import numpy as np

# copied from https://github.com/Javicadserres/quantdare_posts/blob/master/rnn/RNN_utils.py

class Tanh_np:
    """
    activatino tanh with np
    """
    def __init__(self):
        self.out = None

    def forward(self, x:np.array) -> np.array:
        """
        forward process of tanh
        
        Args:
            x (np.array): [batch, ... ]

        Returns:
            np.array: [batch, ... ]
        """
        self.out = np.tanh(x)

        return self.out

    def backward(self, d_prev:np.array) -> np.array:
        """
        backward process of tanh
        
        Args:
            d_prev (np.array): [batch, ... ]

        Returns:
            np.array: [batch, ... ]
        """
        d_prev = d_prev * (1 - np.power(self.out, 2))

        return d_prev
    
    def __call__(self,x):
        return self.forward(x)