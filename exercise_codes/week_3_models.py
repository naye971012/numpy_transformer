from typing import Any
import numpy as np
import torch
import torch.nn as nn
import sys
import os

############## this block is just for import moudles ######
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)
###########################################################

from torch_models.activations.relu import Relu_th
from numpy_models.activations.relu import Relu_np

from torch_models.activations.sigmoid import Sigmoid_th
from numpy_models.activations.sigmoid import Sigmoid_np

from torch_models.losses.binary_ce import Binary_Cross_Entropy_th
from numpy_models.losses.binary_ce import Binary_Cross_Entropy_np

from torch_models.commons.linear import Linear_th
from numpy_models.commons.linear import Linear_np

from numpy_models.losses.ce import Cross_Entropy_np

from numpy_models.commons.cnn import Conv2d_np

from numpy_models.optimizer.SGD_momentum import SGD_momentum_np
from numpy_models.optimizer.SGD import SGD_np
from numpy_models.optimizer.Adam import Adam_np

class model_Adam_np():
    def __init__(self, input_channel=28*28, output_channel=10) -> None:
        
        self.optimizer = Adam_np()
        
        self.linear_1 = Linear_np(input_channel , 256)
        self.linear_2 = Linear_np(256, 128)
        self.linear_3 = Linear_np(128, output_channel)
        self.activation_1 = Relu_np()
        self.activation_2 = Relu_np()
        self.sigmoid = Sigmoid_np()
        
        self.criterion = Cross_Entropy_np()
        
    def forward(self,x):
        
        #make x flatten [# of batch, 28*28 ]
        batch_size = x.shape[0]
        x = x.reshape(batch_size,-1)
        
        x = self.linear_1(x)
        x = self.activation_1(x)
        x = self.linear_2(x)
        x = self.activation_2(x)
        x = self.linear_3(x)
        x = self.sigmoid(x)
        
        return x
    
    def loss(self,x,y):
        loss = self.criterion(x,y)
        return loss
    
    def backward(self):
        d_prev = 1
        d_prev = self.criterion.backward(d_prev)
        d_prev = self.sigmoid.backward(d_prev)
        d_prev = self.linear_3.backward(d_prev)
        d_prev = self.activation_2.backward(d_prev)
        d_prev = self.linear_2.backward(d_prev)
        d_prev = self.activation_1.backward(d_prev)
        d_prev = self.linear_1.backward(d_prev)
    
    def update_grad(self, learning_rate, batch_size):
        self.optimizer.update_grad('linear_3',self.linear_3,learning_rate/batch_size)
        self.optimizer.update_grad('linear_2',self.linear_2,learning_rate/batch_size)
        self.optimizer.update_grad('linear_1',self.linear_1,learning_rate/batch_size)
        
        self.optimizer.step()

class model_SGD_momentum_np():
    def __init__(self, input_channel=28*28, output_channel=10) -> None:
        
        self.optimizer = SGD_momentum_np()
        
        self.linear_1 = Linear_np(input_channel , 256)
        self.linear_2 = Linear_np(256, 128)
        self.linear_3 = Linear_np(128, output_channel)
        self.activation_1 = Relu_np()
        self.activation_2 = Relu_np()
        self.sigmoid = Sigmoid_np()
        
        self.criterion = Cross_Entropy_np()
        
    def forward(self,x):
        
        #make x flatten [# of batch, 28*28 ]
        batch_size = x.shape[0]
        x = x.reshape(batch_size,-1)
        
        x = self.linear_1(x)
        x = self.activation_1(x)
        x = self.linear_2(x)
        x = self.activation_2(x)
        x = self.linear_3(x)
        x = self.sigmoid(x)
        
        return x
    
    def loss(self,x,y):
        loss = self.criterion(x,y)
        return loss
    
    def backward(self):
        d_prev = 1
        d_prev = self.criterion.backward(d_prev)
        d_prev = self.sigmoid.backward(d_prev)
        d_prev = self.linear_3.backward(d_prev)
        d_prev = self.activation_2.backward(d_prev)
        d_prev = self.linear_2.backward(d_prev)
        d_prev = self.activation_1.backward(d_prev)
        d_prev = self.linear_1.backward(d_prev)
    
    def update_grad(self, learning_rate, batch_size):
        self.optimizer.update_grad('linear_3',self.linear_3,learning_rate/batch_size)
        self.optimizer.update_grad('linear_2',self.linear_2,learning_rate/batch_size)
        self.optimizer.update_grad('linear_1',self.linear_1,learning_rate/batch_size)

class model_SGD_np():
    def __init__(self, input_channel=28*28, output_channel=10) -> None:
        
        self.optimizer = SGD_np()
        
        self.linear_1 = Linear_np(input_channel , 256)
        self.linear_2 = Linear_np(256, 128)
        self.linear_3 = Linear_np(128, output_channel)
        self.activation_1 = Relu_np()
        self.activation_2 = Relu_np()
        self.sigmoid = Sigmoid_np()
        
        self.criterion = Cross_Entropy_np()
        
    def forward(self,x):
        
        #make x flatten [# of batch, 28*28 ]
        batch_size = x.shape[0]
        x = x.reshape(batch_size,-1)
        
        x = self.linear_1(x)
        x = self.activation_1(x)
        x = self.linear_2(x)
        x = self.activation_2(x)
        x = self.linear_3(x)
        x = self.sigmoid(x)
        
        return x
    
    def loss(self,x,y):
        loss = self.criterion(x,y)
        return loss
    
    def backward(self):
        d_prev = 1
        d_prev = self.criterion.backward(d_prev)
        d_prev = self.sigmoid.backward(d_prev)
        d_prev = self.linear_3.backward(d_prev)
        d_prev = self.activation_2.backward(d_prev)
        d_prev = self.linear_2.backward(d_prev)
        d_prev = self.activation_1.backward(d_prev)
        d_prev = self.linear_1.backward(d_prev)
    
    def update_grad(self, learning_rate, batch_size):
        self.optimizer.update_grad('linear_3',self.linear_3,learning_rate/batch_size)
        self.optimizer.update_grad('linear_2',self.linear_2,learning_rate/batch_size)
        self.optimizer.update_grad('linear_1',self.linear_1,learning_rate/batch_size)