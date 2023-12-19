import numpy as np

class Cross_Entropy_np():
    def __init__(self,eps=1e-10) -> None:
        self.eps = eps
        self.pred = None # save output
        self.target = None # save target
        self.grad = None # save gradient
    
    def forward(self, pred:np.array, target:np.array ):
        """
        assert pred is prob, not logit!!!
        ( prob = [0,1] , log = [-inf,+inf] )
        """
        if pred.ndim==2:
            """
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
        else:
            #when dim==3
            """
            Args:
                pred: [# of batch, max_len, # of class]
                target: [# of batch, max_len]

            Returns:
                np.array: Scalar
            """
            batch_size, max_len, num_class = pred.shape[0], pred.shape[1], pred.shape[2]

            # target을 one-hot 형태로 변환
            target_one_hot = np.zeros((batch_size, max_len, num_class))
            target_one_hot[np.arange(batch_size)[:, None], np.arange(max_len)[None, :], target] = 1

            output = -1 * (target_one_hot * np.log(pred + self.eps))
            
            self.pred = pred
            self.target = target_one_hot
            
            return np.sum(np.mean(output, axis=(0,1)),axis=0)
    
    def backward(self, d_prev=1):
        """
        Returns:
            np.array: same as pred(input).shape
        """
        
        #divide grad by [# of class] since we apply np.mean in forward
        grad = (self.pred - self.target) / ( (self.pred ) * (1 - self.pred ) + self.eps ) / self.target.shape[-1]
        
        self.grad = grad
        return self.grad
    
    def __call__(self, pred, target):
        return self.forward(pred, target)