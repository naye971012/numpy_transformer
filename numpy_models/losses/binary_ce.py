import numpy as np

class Binary_Cross_Entropy_np():
    def __init__(self,eps=1e-10) -> None:
        self.eps = eps
        self.pred = None # save output
        self.target = None # save target
        self.grad = None # save gradient
    
    def forward(self, pred, target ):
        """
        assert pred is prob, not logit!!!
        ( prob = [0,1] , log = [-inf,+inf] )
        
        Args:
            x (Tensor): [batch, ... ]

        Returns:
            Tensor: Scaler 
        """

        output = -1 * ( target * np.log( pred + self.eps ) + (1-target) * np.log( 1-pred + self.eps ))
        
        self.pred = pred
        self.target = target
        return np.mean(output, axis=None)
    
    def backward(self, d_prev=1):
        """
        backward:
            output -> d_sigmoid -> grad
        
        """

        #divide grad by [# of class] since we apply np.mean in forward
        grad = (self.pred - self.target) / ( (self.pred ) * (1 - self.pred ) + self.eps ) / self.target.shape[-1]
        
        self.grad = grad
        return self.grad
    
    def __call__(self, pred, target):
        return self.forward(pred, target)