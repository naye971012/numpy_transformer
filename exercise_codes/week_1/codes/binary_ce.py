import numpy as np

class Binary_Cross_Entropy_np():
    def __init__(self,eps: int=1e-10) -> None:
        self.eps = eps
        self.pred = None # save output
        self.target = None # save target
        self.grad = None # save gradient
    
    def forward(self, pred:np.array, target:np.array ) -> np.array:
        """
        assert pred is prob, not logit!!!
        ( prob = [0,1] , log = [-inf,+inf] )
        
        Args:
            pred (np.array): [batch, 1]
            target (np.array): [batch]

        Returns:
            np.array: Scaler 
        """
        ############### edit here ###############
        
        output = None
        
        #########################################
        
        self.pred = pred
        self.target = target
        return np.mean(output, axis=None)
    
    def backward(self, d_prev:int=1) -> np.array:
        """
        Returns:
            np.array: same as pred(input).shape
        """

        ############### edit here ###############
        
        grad = None
        
        #########################################
        
        self.grad = grad
        return self.grad
    
    def __call__(self, pred, target):
        return self.forward(pred, target)