import numpy as np

class Batch_Normalization_1D_np:
    def __init__(self, num_features:int,
                       eps:float = 1e-5,
                       momentum: float = 0.9):
        #input is [# of batch, # of feature]
        self.params = dict()
        self.grads = dict() 
        
        self.num_features = num_features 
        self.eps = eps
        self.momentum = momentum
        
        self.running_batch_var = None #for inference
        self.running_batch_mu = None #for inference 
        
        
        self.params['W'] = np.ones((1, num_features)) #init params
        self.params['b'] = np.zeros((1, num_features)) #init params
        
        self.grads['dW'] = None #init grad
        self.grads['db'] = None #init grad
        
        self.flag = 0
        
    
    def save_train_mu_var(self):
        #first init
        if not self.flag and self.running_batch_mu == None:
            self.running_batch_mu = self.batch_mu
            self.running_batch_var = self.batch_var
            self.flag = 1
        #update average running batch mu/var by momentum
        else:
            momentum = self.momentum
            self.running_batch_mu = momentum * self.running_batch_mu + \
                                  (1. - momentum) * self.batch_mu
            self.running_batch_var = momentum * self.running_batch_var + \
                                 (1. - momentum) * self.batch_var
    
    def forward(self, x:np.array, train:bool=True):
        #x.shape = [# of batch, num_features]
        self.num_batch = x.shape[0] # [# of batch]
        
        if train:
            self.batch_mu  = np.mean( x, axis=0 )  # [# of feat]
            self.batch_var =  np.var( x , axis=0 ) # [# of feat]
            self.save_train_mu_var()
        else:
            self.batch_mu = self.running_batch_mu
            self.batch_var = self.running_batch_var

        
        self.batch_var += self.eps
        self.batch_std = np.sqrt(self.batch_var) #[# of feat]
        self.x_minus_mean = x - self.batch_mu #[# of batch, # of feat]
        self.standard_x = self.x_minus_mean / self.batch_std #[# of batch, # of feat]

        self.output = self.params['W'] * self.standard_x + self.params['b'] # [1, # of feature] * [# of batch, # of feature] + [1, # of feature]
        # == [# of batch, # of feature]
        return self.output
        
    def backward(self, d_prev):
        standard_grad = d_prev * self.params['W'] #[1, # of feature]

        var_grad = np.sum(standard_grad * self.x_minus_mean * -0.5 * self.batch_var ** (-3/2),
                          axis=0, keepdims=True) #[# of feat]
        stddev_inv = 1 / self.batch_std  #[# of feat]
        aux_x_minus_mean = 2 * self.x_minus_mean / self.num_batch #[# of batch, # of feature]

        mean_grad = (np.sum(standard_grad * -stddev_inv, axis=0,
                            keepdims=True) +
                            var_grad * np.sum(-aux_x_minus_mean, axis=0,
                            keepdims=True)) #[# of feat]

        self.grads['dW'] = np.sum(d_prev * self.standard_x, axis=0,
                                 keepdims=True) # [1, # of feature]
        self.grads['db'] = np.sum(d_prev, axis=0, keepdims=True) #[ 1, # of feature]

        return standard_grad * stddev_inv + var_grad * aux_x_minus_mean + \
               mean_grad / self.num_batch #[# of batch, # of feature]

    def __call__(self, x):
        return self.forward(x)

if __name__=="__main__":
    # for check dimension
    model = Batch_Normalization_1D_np(50)
    x = np.random.rand(10,50) * 5
    y = model.forward(x)
    print(np.mean(y[0]), np.var(y[0])) #not
    print(np.mean(y[:,0]), np.var(y[:,0])) # N(0,1)
    print(y.shape)
    d_y = model.backward(y)
    print(d_y.shape)