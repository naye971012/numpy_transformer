import numpy as np
import sys
import os

############## this block is just for import moudles ######
current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)
###########################################################

from activations.softmax import softmax_np

#X_Q = [# of batch, query dim, embed dim_xq]
#W_Q = [embed dim_xq, embed dim]

#X_K = [# of batch, key(value) dim, embed dim_xk]
#W_K = [embed dim_xk, embed dim]

#X_V = [# of batch, value dim, embed dim_xv]
#W_V = [embed dim_xv, embed dim]

#Q = [# of batch, query dim, embed dim]
#K = [# of batch, value dim, embed dim]
#V = [# of batch, value dim, embed dim]

#Q@K = [# of batch, query dim, value dim]
#softmax(Q@K) = [# of batch, query dim, value dim]

#V@softmax(Q@K) = [# of batch, query dim, embed dim] = AttentionOutput

#Concat[AttentionOutput] = [# of batch, query dim, embed dim * N ]

class Attention_np:
    """
    attention layer which get query,key,value as input \
    input dimension should be [# of batch, channel 1, cnahhel 2] \
    currently, do not support CNN attention \
    this attention layer doesn't have WX+b in final output
    """
    def __init__(self, query_embed_dim:int, 
                 key_embed_dim:int, 
                 value_embed_dim:int, 
                 attention_embed_dim:int = 256
                 ) -> None:
        self.params = dict()
        self.grads = dict()
        
        self.softmax = softmax_np()
        
        self.q_dim = query_embed_dim
        self.k_dim = key_embed_dim
        self.v_dim = value_embed_dim
        self.embed_dim = attention_embed_dim

        limit = np.sqrt(2 / float(self.q_dim))
        self.params['W_Q'] = np.random.normal(0.0, limit, size=( self.q_dim, self.embed_dim))
        self.params['b_Q'] = np.zeros(shape = ( self.embed_dim , ) )

        limit = np.sqrt(2 / float(self.k_dim))
        self.params['W_K'] = np.random.normal(0.0, limit, size=( self.k_dim, self.embed_dim))
        self.params['b_K'] = np.zeros(shape = ( self.embed_dim , ) )

        limit = np.sqrt(2 / float(self.v_dim))
        self.params['W_V'] = np.random.normal(0.0, limit, size=( self.v_dim, self.embed_dim))
        self.params['b_V'] = np.zeros(shape = ( self.embed_dim , ) )
        
    def forward(self,
                x_q:np.array=0,
                x_k:np.array=0,
                x_v:np.array=0):
        """
        #X_Q = [# of batch, query dim, embed dim_xq]
        #X_K = [# of batch, key(value) dim, embed dim_xk]
        #X_V = [# of batch, value dim, embed dim_xv]
        
        return: [# of batch, query dim, embed dim]
        """
        self.x_q = x_q
        self.x_k = x_k
        self.x_v = x_v

        #Q = [# of batch, query dim, embed dim]
        #K = [# of batch, value dim, embed dim]
        #V = [# of batch, value dim, embed dim]
        self.Q = np.dot(x_q,self.params['W_Q']) + self.params['b_Q']
        self.K = np.dot(x_k,self.params['W_K']) + self.params['b_K']
        self.V = np.dot(x_v,self.params['W_V']) + self.params['b_V']

        #Q@K = [# of batch, query dim, value dim]
        #softmax(Q@K) = [# of batch, query dim, value dim]
        #V@softmax(Q@K) = [# of batch, query dim, embed dim] = AttentionOutput
        self.QK = np.matmul(self.Q, np.transpose(self.K, (0, 2, 1)))
        self.softQK = self.softmax(self.QK)
        self.VsoftQK = np.matmul(self.V.transpose(0, 2, 1), self.softQK.transpose(0, 2, 1)).transpose(0, 2, 1)
        
        return self.VsoftQK
        
        
    def backward(self, d_prev):
        """
        d_prev = [# of batch, query dim, embed dim]
        
        output = [# of batch, query dim, embed dim_xq]
        """
        pass
        #self.grads['dW_Q'] = grad_W_Q
        #self.grads['db_Q'] = grad_b_Q
        
        #self.grads['dW_K'] = grad_W_K
        #self.grads['db_K'] = grad_b_K
        
        #self.grads['dW_V'] = grad_W_V
        #self.grads['db_V'] = grad_b_V
        
        #return d_Q
    
    def __call__(self,*args):
        return self.forward(*args)


#test forward/backward dimension
if __name__ == "__main__":
    model = Attention_np(10,10,10,40)
    q = np.random.randn(3,5,10)
    kv = np.random.randn(3,5,10)
    
    output = model(q,kv,kv)
    print(output.shape)
    
    model.backward(output)