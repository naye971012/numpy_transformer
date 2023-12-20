import numpy as np
import random

from codes.fixed.linear import Linear_np
from codes.fixed.positional_encoding import Embedding_with_positional_encoding_np
from codes.fixed.embedding import Embedding_np
from codes.fixed.flatten import Flatten_np
from codes.fixed.SGD import SGD_np
from codes.fixed.ce import Cross_Entropy_np
from codes.fixed.binary_ce import Binary_Cross_Entropy_np
from codes.fixed.sigmoid import Sigmoid_np
from codes.fixed.softmax import softmax_np
from codes.fixed.relu import Relu_np
from codes.fixed.layernorm import Layer_Normalization_np

from codes.attention import Attention_np

class model_with_attention():
    def __init__(self, input_channel=10, output_channel=10) -> None:
        
        self.optimizer = SGD_np()
        
        # [# of batch, 10] -> [# of batch, 10, 32]
        self.embedding = Embedding_with_positional_encoding_np(10,256)
        
        # [# of batch, 10, 32] -> [# of batch, 10, 16]
        self.norm1 = Layer_Normalization_np()
        self.attention = Attention_np(256,256,256,256)
        
        self.linear_1 = Linear_np(256 , 512)
        self.norm2 = Layer_Normalization_np()
        self.linear_2 = Linear_np(512, output_channel)
        
        self.activation_1 = Relu_np()
        self.activation_2 = Relu_np()
        self.softmax = softmax_np()
        
        self.flatten = Flatten_np()
        
        self.criterion = Cross_Entropy_np()
        
    def forward(self,x):   
        #x[# of batch, 10]
        
        #x[# of batch, 10, 32]
        x = self.embedding(x)
        
        # value of 0~9, query
        q = self.embedding( np.tile( np.arange(10), (x.shape[0],1) ) )
        
        #x[# of batch, 10 16]
        x, att_map = self.attention(q,x,x) #query key value
        x = self.norm1(x)
        x = self.activation_1(x)
        
        x = self.linear_1(x)
        x = self.norm2(x)
        x = self.activation_2(x)
        
        x = self.linear_2(x) #[# of batch, 10, 2]
        x = self.softmax(x)
        
        # [# of batch, 10]
        #x = self.flatten(x)
        
        return x, att_map
    
    def loss(self,x,y):
        loss = self.criterion(x,y)
        return loss
    
    def backward(self,d_prev=1):
        d_prev = self.criterion.backward(d_prev)
        #d_prev = self.flatten.backward(d_prev)
        
        d_prev = self.softmax.backward(d_prev)
        d_prev = self.linear_2.backward(d_prev)
        
        d_prev = self.activation_2.backward(d_prev)
        d_prev = self.norm2.backward(d_prev)
        d_prev = self.linear_1.backward(d_prev)
        
        d_prev = self.activation_1.backward(d_prev)
        d_prev = self.norm1.backward(d_prev)
        _ , d_prev = self.attention.backward(d_prev)
        
        d_prev = self.embedding.backward(d_prev)
        return d_prev
    
    def update_grad(self, learning_rate, batch_size):
        self.optimizer.update_grad('emb',self.embedding,learning_rate/batch_size)
        self.optimizer.update_grad('att',self.attention,learning_rate/batch_size)
        self.optimizer.update_grad('linear_2',self.linear_2,learning_rate/batch_size)
        self.optimizer.update_grad('linear_1',self.linear_1,learning_rate/batch_size)
        
        self.optimizer.step()

class CustomDataset:
    def __init__(self,make_len:int=10) -> None:
        #max sentence len
        self.max_len = make_len
    
    def make_data(self):
        
        data = np.arange(10)
        np.random.shuffle(data)
        
        #data = np.array([random.randint(0, 9) for _ in range(self.max_len)])
        #label = np.array([1 if (data[(i+1)%10]==data[i] or data[(i-1)%10]==data[i]) else 0 for i in range(len(data))])
        label = np.array([ i+1 if data[i]==i else 0 for i in range(len(data)) ])
        
        return data,label
    
    def __getitem__(self,x):
        return self.make_data()
    
    def __len__(self):
        return 1000

class CustomDataloader:
    """
        my custom dataloader class
        if len(dataset)%batch_size !=0, ignore remain items
    """
    def __init__(self, dataset, batch_size:int, shuffle:bool) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __getitem__(self, idx: int):
        output = {
            "data" : [],
            "label" : []
        }
        for i in range(self.batch_size * idx , self.batch_size * (idx+1), 1 ):
            _input, _output = self.dataset[i]
            output["data"].append(_input)
            output["label"].append(_output)

        return output
    
    def __len__(self) ->int:
        return len(self.dataset) // self.batch_size
