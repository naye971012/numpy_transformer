import numpy as np

# copied from https://github.com/Javicadserres/quantdare_posts/blob/master/rnn/RNN_utils.py

class Tanh_np:
    """
    Class for the Hyperbolic tangent activation function.
    Applies the hyperbolic tangent function:
    :math:`Tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}`  
    """
    def __init__(self):
        self.type = 'Tanh'

    def forward(self, Z):
        """
        Computes the forward propagation.
        Parameters
        ----------
        Z : numpy.array
            Input.
        Returns
        -------
        A : numpy.array
            Output.
        """
        self.A = np.tanh(Z)

        return self.A

    def backward(self, dA):
        """
        Computes the backward propagation.
        Parameters
        ----------
        dA : numpy.array
            Gradients of the activation function output.
        Returns
        -------
        dZ : numpy.array
            Gradients of the activation function input.
        """
        dZ = dA * (1 - np.power(self.A, 2))

        return dZ
    
    def __call__(self,x):
        return self.forward(x)