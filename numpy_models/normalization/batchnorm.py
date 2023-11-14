import numpy as np

import torch
batchnorm1d =torch.nn.BatchNorm1d()
batchnorm2d =torch.nn.BatchNorm2d()

class Batch_Normalization_1D_np:
    def __init__(self, num_features:int,
                       eps:float = 1e-5,
                       train:object = 'train'):
        self.num_features = num_features
        self.eps = eps
        self.train = True if (train=='train' or train==True) else False
        
        self.gamma = np.ones((1, num_features)) #init params
        self.bias = np.zeros((1, num_features)) #init params
        
        self.gamma_grad = None #init grad
        self.bias_grad = None #init grad
        
        self.params [self.gamma, self.bias] #for optimizer update
        self.grads = [self.gamma_grad, self.bias] #for optimizer update
        
    def forward(self, x):
        #x.shape = [# of batch, layer 1]
        self.num_batch = x.shape[0] # [# of batch]
        
        self.batch_mu  = np.mean( np.sum(x,axis=0) ) #scaler value
        self.batch_var =  np.var( np.sum(x,axis=0) ) #scaler value
        self.x_hat = (x - self.batch_mu) / np.sqrt(self.batch_var + self.eps) # [# of batch, layer 1]

        self.output = self.gamma * self.x_hat + self.bias
        return self.output
        
    def backward(self, d_prev):
        pass

    def __call__(self, x):
        return self.forward(x)