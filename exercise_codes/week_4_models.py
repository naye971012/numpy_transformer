import numpy as np
from typing import Any, Tuple,List, Dict
import os
import sys

############## this block is just for import moudles ######
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)
###########################################################

from numpy_models.tokenizer.vocab import Vocabulary
from numpy_models.tokenizer.word_tokenizer import Word_tokenizer_np

from numpy_models.utils.embedding import Embedding_np

from numpy_models.commons.linear import Linear_np
from numpy_models.commons.rnn import RNN_np

from numpy_models.optimizer.Adam import Adam_np

from numpy_models.losses.ce import Cross_Entropy_np

from numpy_models.activations.softmax import softmax_np
from numpy_models.activations.relu import Relu_np

class myModel:
    def __init__(self, tokenizer: Word_tokenizer_np) -> None:
        
        self.criterion = Cross_Entropy_np()
        self.optimizer = Adam_np()
        
        self.tokenizer = tokenizer
        
        self.embedding = Embedding_np(len(self.tokenizer),300)
        
        self.rnn1 = RNN_np(300,400,2)
        
        self.activation1 = Relu_np()
        self.linear1 = Linear_np(400,300)
        
        self.activation2 = Relu_np()
        self.linear2 = Linear_np(300,len(self.tokenizer))
        
        self.softmax = softmax_np()
        
    def forward(self,x: List[str]):
        #input : [# of batch, string sentence]
        
        #x: [# of batch, words]
        x = self.tokenizer.tokenize(x)
        
        #x: [# of batch, max_length]
        x = self.tokenizer.convert_tokens_to_ids(x, padding=True, max_length=20)
        
        #x: [# of batch, max_length, # of embedding(300)]
        x = self.embedding(x)
        
        #x: [# of batch, max_length, 400]
        x = self.rnn1(x)
        
        #x: [# of batch, max_length, 400] with logit
        x = self.activation1(x)
        x = self.linear1(x)
        
        #x: [# of batch, max_length, 300] with logit
        x = self.activation2(x)
        x = self.linear2(x)
        
        #x: [# of batch, max_length, 300] with prob
        x = self.softmax(x)
        
        return x
        
    def backward(self,d_prev=1):
        d_prev = self.criterion.backward(d_prev)
        d_prev = self.softmax.backward(d_prev)
        
        d_prev = self.linear2.backward(d_prev)
        d_prev = self.activation2.backward(d_prev)
        d_prev = self.linear1.backward(d_prev)
        d_prev = self.activation1.backward(d_prev)
        
        d_prev = self.rnn1.backward(d_prev)
        
        d_prev = self.embedding.backward(d_prev)
        
        return d_prev
    
    
    def loss(self, pred: np.array , real_txt: List[str] ):
        #pred shape = [# of batch, max_length, vocab size]
        #real shape = [# of batch, max_length]
        
        real = self.tokenizer.tokenize(real_txt)
        real = self.tokenizer.convert_tokens_to_ids(real, padding=True, max_length=20)
        
        _loss = self.criterion(pred, real)
        return _loss


    def update_grad(self, learning_rate, batch_size):
        
        self.optimizer.update_grad('rnn1', self.rnn1, learning_rate/batch_size)
        self.optimizer.update_grad('linear_2', self.linear2, learning_rate/batch_size)
        self.optimizer.update_grad('linear_1', self.linear1, learning_rate/batch_size)
        self.optimizer.update_grad('embedding', self.embedding, learning_rate/batch_size)
        
        self.optimizer.step()

    def predict(self, text:str):
        output = self.forward([text])[0]
        output_ids = np.argmax(output,axis=1)
        output_txt = self.tokenizer.convert_one_ids_to_tokens(output_ids)
        return output_txt

    def __call__(self, x):
        return self.forward(x)
        
        

class CustomDataset:
    def __init__(self, chat_list: List[str]) -> None:
        self.chat_list = chat_list
    
    def __len__(self) -> int:
        return len(self.chat_list) - 1
    
    def __getitem__(self,idx: int) -> Tuple[str, str]:
        chatting_input = self.chat_list[idx]
        chatting_output = self.chat_list[idx+1]
        
        return (chatting_input, chatting_output)

class CustomDataloader:
    """
        my custom dataloader class
        if len(dataset)%batch_size !=0, ignore remain items
    """
    def __init__(self, dataset, batch_size:int, shuffle:bool) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __getitem__(self, idx: int) -> Dict:
        output = {
            "input" : [],
            "output" : []
        }
        for i in range(self.batch_size * idx , self.batch_size * (idx+1), 1 ):
            _input, _output = self.dataset[i]
            output["input"].append(_input)
            output["output"].append(_output)

        return output
    
    def __len__(self) ->int:
        return len(self.dataset) // self.batch_size

