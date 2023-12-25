from typing import *
import sys
import os
import numpy as np

############## this block is just for import moudles ######
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
grand_path = os.path.dirname(parent_path)
sys.path.append(grand_path)
###########################################################

from numpy_functions import *

"""
Simple GAN model
training based on fashion_mnist dataset
"""

class Discriminator_np:
    def __init__(self, 
                 img_size:Tuple[int,int]=(28,28), 
                 img_channel:int=1) -> None:
        """
        based on fahsion_manist, which img size is 28,28 and channel is 1
        """

        self.w, self.h = img_size
        self.c = img_channel
        
        self.init_param()
    
    def init_param(self):
        
        init_dim = self.c * self.w * self.h
        
        self.linear1 = Linear_np(init_dim, init_dim*8)
        self.layernorm1 = Layer_Normalization_np()
        self.acti1 = Relu_np()
        self.drop1 = Dropout_np(prob=0.2)
        
        self.linear2 = Linear_np(init_dim*8, init_dim*8)
        self.layernorm2 = Layer_Normalization_np()
        self.acti2 = Relu_np()
        self.drop2 = Dropout_np(prob=0.2)
        
        self.linear3 = Linear_np(init_dim*8, 1)
        
        self.flatten = Flatten_np()
        
        self.sigmoid = Sigmoid_np()
    
        self.criterion = Binary_Cross_Entropy_np()
    
    def forward(self, x:np.array) -> np.array:
        """
        input: [batch, 1,28,28]
        output: binary classification
        """
        x = self.flatten(x)
        
        x = self.linear1(x)
        x = self.layernorm1(x)
        x = self.acti1(x)
        x = self.drop1(x)
        
        x = self.linear2(x)
        x = self.layernorm2(x)
        x = self.acti2(x)
        x = self.drop2(x)
        
        x = self.linear3(x)
        
        x = self.sigmoid(x)
        
        x = x.reshape(-1) # make [batch, 1] -> [batch] for bce loss
        return x
    
    def loss(self,
             pred:np.array,
             label:np.array) -> np.array:
        return self.criterion(pred,label)
    
    def backward(self, d_prev=1) -> np.array:
       
        d_prev = self.criterion.backward(d_prev=1)
        
        d_prev = self.sigmoid.backward(d_prev)
        d_prev = self.linear3.backward(d_prev)
        
        d_prev = self.drop2.backward(d_prev)
        d_prev = self.acti2.backward(d_prev)
        d_prev = self.layernorm2.backward(d_prev)
        d_prev = self.linear2.backward(d_prev)
        
        d_prev = self.drop1.backward(d_prev)
        d_prev = self.acti1.backward(d_prev)
        d_prev = self.layernorm1.backward(d_prev)
        d_prev = self.linear1.backward(d_prev)
        
        d_prev = self.flatten.backward(d_prev)

        return d_prev
    
    def __call__(self, x:np.array) -> np.array:
        return self.forward(x)

    def update_grad(self,
                    name:str="1",
                    optimizer:Any=SGD_np,
                    LR:float=0.001):
        """
        update weight recursively
        
        if layer is in numpy_functions, update gradient
        else(layer is a block), call update_grad function recursively
        
        Args:
            name (str): distinguish value
            optimizer (Any): your optimizer
            lr (float): learning rate
        """
        
        optimizer.update_grad('discriminator_linear1'+name, self.linear1, LR)
        optimizer.update_grad('discriminator_linear2'+name, self.linear2, LR)
        optimizer.update_grad('discriminator_linear3'+name, self.linear3, LR)



class Generator_np:
    def __init__(self,b,c,w,h) -> None:
        self.b=b
        self.c=c
        self.w=w
        self.h=h

        self.init_param()
        
    def init_param(self):
        init_dim = self.c*self.w*self.h
        
        self.linear1 = Linear_np(init_dim, init_dim*8)
        self.layernorm1 = Layer_Normalization_np()
        self.acti1 = Relu_np()
        self.drop1 = Dropout_np(prob=0.2)
        
        self.linear2 = Linear_np(init_dim*8, init_dim*8)
        self.layernorm2 = Layer_Normalization_np()
        self.acti2 = Relu_np()
        self.drop2 = Dropout_np(prob=0.2)
        
        self.linear3 = Linear_np(init_dim*8, init_dim)
        self.sigmoid = Sigmoid_np()
    
    def make_normal_distribution(self) -> np.array:
        
        arr = np.random.normal(0,1,size=(self.b,self.c*self.w*self.h))
        return arr
    
    def forward(self, x:np.array) -> np.array:
        """
        input x: normal distribution array [# of batch, c*w*h ]
        output x: image shape [# of batch, c, w, h]
        """
        
        x = self.linear1(x)
        x = self.layernorm1(x)
        x = self.acti1(x)
        x = self.drop1(x)
        
        x = self.linear2(x)
        x = self.layernorm2(x)
        x = self.acti2(x)
        x = self.drop2(x)
        
        x = self.linear3(x)
        x = self.sigmoid(x)
        
        x = x.reshape(self.b,self.c,self.w,self.h)
        
        return x
    
    def backward(self, d_prev) -> None:
        """
        input x: image shape [# of batch, c, w, h]
        """
        d_prev = d_prev.reshape(self.b,-1)
        
        d_prev = self.sigmoid.backward(d_prev)
        d_prev = self.linear3.backward(d_prev)
        
        d_prev = self.drop2.backward(d_prev)
        d_prev = self.acti2.backward(d_prev)
        d_prev = self.layernorm2.backward(d_prev)
        d_prev = self.linear2.backward(d_prev)
        
        d_prev = self.drop1.backward(d_prev)
        d_prev = self.acti1.backward(d_prev)
        d_prev = self.layernorm1.backward(d_prev)
        d_prev = self.linear1.backward(d_prev)
    
    def __call__(self, x:np.array) -> np.array:
        return self.forward(x)
    
    def update_grad(self,
                    name:str="1",
                    optimizer:Any=SGD_np,
                    LR:float=0.001):
        """
        update weight recursively
        
        if layer is in numpy_functions, update gradient
        else(layer is a block), call update_grad function recursively
        
        Args:
            name (str): distinguish value
            optimizer (Any): your optimizer
            lr (float): learning rate
        """
        
        optimizer.update_grad('generator_linear1'+name, self.linear1, LR)
        optimizer.update_grad('generator_linear2'+name, self.linear2, LR)
        optimizer.update_grad('generator_linear3'+name, self.linear3, LR)