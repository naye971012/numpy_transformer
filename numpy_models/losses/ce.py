import numpy as np

class Cross_Entropy_np():
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
            pred: [# of batch, # of class]
            target: [# of batch]

        Returns:
            Tensor: Scaler 
        """
        batch_size, num_class = pred.shape[0], pred.shape[1]
        target_one_hot = np.zeros((batch_size, num_class))
        target_one_hot[np.arange(batch_size), target] = 1
        
        output = -1 * (target_one_hot * np.log( pred + self.eps) ) # sigma y_target * np.log(y_pred)
        
        self.pred = pred
        self.target = target_one_hot
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