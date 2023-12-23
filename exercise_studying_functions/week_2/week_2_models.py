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

from torch.nn import ReLU as Relu_th
from torch.nn import Sigmoid as Sigmoid_th
from torch.nn import Linear as Linear_th

from codes.fixed.relu import Relu_np
from codes.fixed.sigmoid import Sigmoid_np
from codes.fixed.linear import Linear_np
from codes.fixed.ce import Cross_Entropy_np

from codes.cnn import Conv2d_np
from codes.pooling import MaxPooling2D_np

class linear_model_th(nn.Module):
    def __init__(self, input_channel=28*28, output_channel=10) -> None:
        super().__init__()
        
        self.linear_1 = Linear_th(input_channel , 256)
        self.linear_2 = Linear_th(256, 128)
        self.linear_3 = Linear_th(128, output_channel)
        self.activation_1 = Relu_th()
        self.activation_2 = Relu_th()
        self.sigmoid = Sigmoid_th()
        
        
    def forward(self,x):
        #make x flatten [# of batch, 28*28 ]
        batch_size = x.size()[0]
        x = x.view(batch_size,-1)
        
        x = self.linear_1(x)
        x = self.activation_1(x)
        x = self.linear_2(x)
        x = self.activation_2(x)
        x = self.linear_3(x)
        x = self.sigmoid(x)
        
        return x

class linear_model_np():
    def __init__(self, input_channel=28*28, output_channel=10) -> None:
        
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
        self.linear_3.params['W'] -= self.linear_3.grads['dW'] * learning_rate / batch_size
        self.linear_3.params['b'] -= self.linear_3.grads['db'] * learning_rate / batch_size
        self.linear_2.params['W'] -= self.linear_2.grads['dW'] * learning_rate / batch_size
        self.linear_2.params['b'] -= self.linear_2.grads['db'] * learning_rate / batch_size
        self.linear_1.params['W'] -= self.linear_1.grads['dW'] * learning_rate / batch_size
        self.linear_1.params['b'] -= self.linear_1.grads['db'] * learning_rate / batch_size
     

class cnn_model_th(nn.Module):
    def __init__(self, input_channel=28*28, output_channel=10) -> None:
        super().__init__()
        
        self.cnn_1 = nn.Conv2d(1 , 8, kernel_size=(3,3), stride=(1,1),padding=(1,1)) #28,28 -> 28,28
        self.pooling1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)) #28,28 -> 14,14
        self.cnn_2 = nn.Conv2d(8 , 12, kernel_size=(3,3), stride=(1,1),padding=(1,1)) #14,14 -> 14,14
        self.pooling2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)) #14,14 -> 7.7
        
        self.linear1 = Linear_th( 7*7*12 , 300)
        self.linear2 = Linear_th(300, output_channel)
        
        self.activation_1 = Relu_th()
        self.activation_2 = Relu_th()
        self.activation_3 = Relu_th()
        self.sigmoid = Sigmoid_th()
        
    
    def forward(self,x):
        #change [# of batch, channel, x_shape, y_shape]
        batch_size = x.size()[0]
        x = x.view(-1,1,28,28)
        
        #cnn part
        x = self.cnn_1(x)
        x = self.activation_1(x)
        x = self.pooling1(x)
        x = self.cnn_2(x)
        x = self.activation_2(x)
        x = self.pooling2(x)
        
        #flatten
        x = x.view(batch_size, -1)
        
        #linear part
        x = self.linear1(x)
        x = self.activation_3(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        
        return x

class cnn_model_np():
    def __init__(self, input_channel=28*28, output_channel=10) -> None:
        
        self.conv_1 = Conv2d_np(1,8)
        self.pooling_1 = MaxPooling2D_np()
        self.conv_2 = Conv2d_np(8,12)
        self.pooling_2 = MaxPooling2D_np()
        self.linear_1 = Linear_np(7*7*12, 300)
        self.linear_2 = Linear_np(300,output_channel)
        self.activation_1 = Relu_np()
        self.activation_2 = Relu_np()
        self.activation_3 = Relu_np()
        self.sigmoid = Sigmoid_np()
        
        self.criterion = Cross_Entropy_np()
        
    def forward(self,x):
        
        batch_size = x.shape[0]
        x = x.reshape(-1,1,28,28)
        
        #cnn part
        
        ############## edit here ##############
        
        
        
        #######################################
        
        #flatten
        x = x.reshape(batch_size, -1)
        
        #linear part
        
        ############## edit here ##############
        
        
        
        #######################################
    
    def loss(self,x,y):
        loss = self.criterion(x,y)
        return loss
    
    
    def backward(self):
        d_prev = 1
        
        ############## edit here ##############
        
        
        pass
        #######################################

    def update_grad(self, learning_rate, batch_size):
        
        ############## edit here ##############
        
        
        pass
        #######################################
        