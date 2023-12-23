from typing import Any
import numpy as np
import copy

class Linear_np:
    """
    Linear layer with numpy
    """
    def __init__(self, num_hidden_1, num_hidden_2):
        
        self.params = dict()
        self.grads = dict()
        
        self.output = None
        self.grad = None
        
        self.hid1 = num_hidden_1
        self.hid2 = num_hidden_2
        
        self.init_params()
    
    def init_params(self):
        """
        init params.
        int linear layer, it has 'W' and 'b'
        
        'W': [num_hidden_1, num_hidden_2]
        'b': [num_hidden_2]
        """
        limit = np.sqrt(2 / float(self.hid1))
        self.params['W'] = np.random.normal(0.0, limit, size=(self.hid1, self.hid2))
        self.params['b'] = np.zeros(self.hid2)

        self.grads['dW'] = None
        self.grads['db'] = None
        
    def forward(self, x):
        
        """
        Linear layer forward
        - Feed forward
        - Apply activation function you implemented above.

        [Inputs]
           x : Input data (N, D)

        [Outputs]
            self.out : Output of Linear layer. Hidden. (N, H)
        """

        self.x = x
        self.out = np.matmul(self.x, self.params['W']) + self.params['b']

        return self.out

    def backward(self, d_prev):
        """
        Linear layer backward
        x and (W & b) --> z -- (activation) --> hidden
        dx and (dW & db) <-- dz <-- (activation) <-- hidden

        - Backward of activation
        - Gradients of W, b

        [Inputs]
            d_prev : Gradients until now. (N,H)

        [Outputs]
            dx : Gradients of input x (==self.grad)
        """
        #### dW
        # x.T = (N,D) -> (D,N)       
        # x.T * d_prev = (D,N) * (N,H) = (D,H)
        
        #### db
        # np.sum(d_prev) = (N,H) -> (H)
        
        #### grad
        # W * d_prev = (D,H) * (N,H) -> (D,N)
        # (D,N).T = (N,D)
        
        if (d_prev.ndim == 2):
            self.grads['dW'] = self.x.T.dot(d_prev) 
            self.grads['db'] = np.sum(d_prev, axis=0)
            self.grad = np.matmul(self.params['W'], d_prev.T).T
        else:
            #when 3 dim
            self.grads['dW'] = np.tensordot(self.x, d_prev, axes=([0, 1], [0, 1]))
            self.grads['db'] = np.sum(d_prev, axis=(0,1) )
            self.grad = np.dot(d_prev , self.params['W'].T)
        
        return self.grad

    def __call__(self, x):
        return self.forward(x)