import numpy as np

class Linear_np:
    def __init__(self, num_hidden_1, num_hidden_2):
        limit = np.sqrt(2 / float(num_hidden_1))
        self.W = np.random.normal(0.0, limit, size=(num_hidden_1, num_hidden_2))
        self.b = np.zeros(num_hidden_2)

        self.output = None
        self.dW = None
        self.db = None
        self.grad = None
        
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
        self.out = np.matmul(self.x, self.W) + self.b

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
        
        self.dW = self.x.T.dot(d_prev) 
        self.db = np.sum(d_prev, axis=0)
        self.grad = np.matmul(self.W, d_prev.T).T
        
        return self.grad

    def __call__(self, x):
        return self.forward(x)