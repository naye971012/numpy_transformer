import numpy as np

class Layer_Normalization_np:
    def __init__(self, eps:float = 1e-5,):
        #input is [# of batch, feat1, feat2, ...]
        self.eps = eps
    
    def forward(self, x:np.array, train:bool=True):
        #x.shape = [# of batch, num_features1, num_features2, ...]
        original_shape = x.shape
        self.num_batch = x.shape[0] # [# of batch]
        x = x.reshape(self.num_batch,-1) #flatten, [# of batch, -1]
        self.x = x
        
        #each features mu and var
        self.feature_mu  = np.mean( x.reshape(self.num_batch,-1), axis=1 ).reshape(-1,1)  # [# of batch, 1]
        self.feature_var =  np.var( x.reshape(self.num_batch,-1), axis=1 ).reshape(-1,1)  # [# of batch, 1]

        #print(self.batch_mu.shape) #[# of batch, 1]
        
        self.feature_var += self.eps
        self.feature_std = np.sqrt(self.feature_var) #[# of batch]
        self.x_minus_mean = x - self.feature_mu # [# of batch, # of feat] - [# of batch, 1] (broadcast)
        self.standard_x = self.x_minus_mean / self.feature_std #[# of batch, # of feat]

        output = self.standard_x.reshape(original_shape) #restore flatten d_prev
        return output #no dimension change(mayebe?)
        
    def backward(self, d_prev):
        #d_prev.shape = [# of batch, feat1, feat2, ...]
        original_shape = d_prev.shape
        d_prev = d_prev.reshape(self.num_batch,-1) #flatten, [# of batch, -1]
        
        mean = self.feature_mu #[# of batch, 1]
        std = self.feature_std #[# of batch, 1]
        
        dx = (1.0 / std) * d_prev # [# of batch, -1]
        dmean = -dx.sum(axis=1, keepdims=True)
        dstd = -np.sum(self.standard_x * dx, axis=1, keepdims=True) / (std) # [# of batch, # of feat]
        
        dx += (1.0 / d_prev.shape[1]) * (self.x - mean) * dstd
        
        dx = dx.reshape(original_shape) #restore flatten d_prev
        return dx

    def __call__(self, x):
        return self.forward(x)

if __name__=="__main__":
    # for check dimension
    model = Layer_Normalization_np()
    x = np.random.rand(5,10) * 5
    y = model.forward(x)
    print(np.mean(y[:,0]), np.var(y[:,0])) #not
    print(np.mean(y[0]), np.var(y[0])) # N(0,1)
    print(y.shape)
    d_y = model.backward(np.random.randn(5,10))
    print(d_y.shape)
    #print(y[0])
    #print(d_y[0])