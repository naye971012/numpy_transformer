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
        
        self.conv1 = Conv2d_np(in_c=self.c,
                                     out_c=self.c*4,
                                     padding=(3,3)) #[1,28,28] -> [4,28,28]
        
        self.pool = 
        
        self.conv2 = Conv2d_np(in_c=self.c*4,
                                     out_c=self.c*8,
                                     padding=(3,3)) #[1,28,28] -> [4,28,28]
        
    
    def forward(self, x:np.array) -> np.array:
        pass
    
    def loss(self,
             pred:np.array,
             label:np.array) -> np.array:
        pass
    
    def backward(self, d_prev=1) -> None:
        pass
    
    def __call__(self, x:np.array) -> np.array:
        pass
    
    def update_grad(self):
        pass



class Generator_np:
    def __init__(self) -> None:
        pass
    
    def init_param(self):
        pass
    
    def forward(self, x:np.array) -> np.array:
        pass
    
    def predict(self, x:np.array) -> np.array:
        pass
    
    def loss(self,
             pred:np.array,
             label:np.array) -> np.array:
        pass
    
    def backward(self, d_prev=1) -> None:
        pass
    
    def __call__(self, x:np.array) -> np.array:
        pass
    
    def update_grad(self):
        pass