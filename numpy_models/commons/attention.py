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
    input dimension should be [# of batch, ..., channel ] \
    this attention layer doesn't have WX+b in final output
    """
    def __init__(self, query_embed_dim:int, 
                 key_embed_dim:int, 
                 value_embed_dim:int, 
                 attention_embed_dim:int = 256,
                 scale = True
                 ) -> None:
        self.params = dict()
        self.grads = dict()
        
        self.scale = scale
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
                x_q:np.array,
                x_k:np.array,
                x_v:np.array):
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
        
        """Actual computation of forward pass"""
        scale = 1 / np.sqrt(self.Q.shape[-1]) if self.scale else 1
        self.QK = self.Q @ self.K.swapaxes(-2, -1) * scale  # attention scores
        self.softQK = self.softmax.forward(self.QK)  # attention weights
        self.VsoftQK = self.softQK @ self.V
        
        #following is old code
        #self.QK = np.matmul(self.Q, np.transpose(self.K, (0, 2, 1)))
        #self.softQK = self.softmax(self.QK)
        #self.VsoftQK = np.matmul(self.V.transpose(0, 2, 1), self.softQK.transpose(0, 2, 1)).transpose(0, 2, 1)
        
        return self.VsoftQK, self.softQK #attention output and attention map
        
        
    def backward(self, d_prev):
        """
        d_prev = [# of batch, query dim, embed dim]
        
        output = [# of batch, query dim, embed dim_xq]
        """
        
        dQ, dK, dV = [], [], []
        weights = self.softQK
        for i, (dy, q, k, v, w) in enumerate(zip(d_prev, self.Q, self.K, self.V, weights)):
            dq, dk, dv = self._bwd(i, dy, q, k, v, w)
            dQ.append(dq)
            dK.append(dk)
            dV.append(dv)

        #if len(self.Q) == 1:
        #    dQ, dK, dV = dQ[0], dK[0], dV[0]

        dQ = np.array(dQ)
        dK = np.array(dK)
        dV = np.array(dV)
        
        
        #X_Q = [# of batch, query dim, embed dim_xq]
        #W_Q = [embed dim_xq, embed dim]

        #X_K = [# of batch, key(value) dim, embed dim_xk]
        #W_K = [embed dim_xk, embed dim]

        #X_V = [# of batch, value dim, embed dim_xv]
        #W_V = [embed dim_xv, embed dim]

        #Q = [# of batch, query dim, embed dim]
        #K = [# of batch, value dim, embed dim]
        #V = [# of batch, value dim, embed dim]
        
        #when 3 dim
        self.grads['dW_Q'] = np.tensordot(self.x_q, dQ, axes=([0, 1], [0, 1]))
        self.grads['db_Q'] = np.sum(dQ, axis=(0,1) )
        self.grad_q = np.dot(dQ , self.params['W_Q'].T)
        #when 3 dim
        self.grads['dW_K'] = np.tensordot(self.x_k, dK, axes=([0, 1], [0, 1]))
        self.grads['db_K'] = np.sum(dK, axis=(0,1) )
        self.grad_k = np.dot(dK , self.params['W_K'].T)
        #when 3 dim
        self.grads['dW_V'] = np.tensordot(self.x_v, dV, axes=([0, 1], [0, 1]))
        self.grads['db_V'] = np.sum(dV, axis=(0,1) )
        self.grad_v = np.dot(dV , self.params['W_V'].T)        
        
        #when q==k==v (self attention)
        if(np.array_equal(self.x_k,self.x_q)):
            return self.grad_q + self.grad_k + self.grad_v
        #when encoder-decoder attention
        else:
            return self.grad_q , self.grad_k + self.grad_v

    def _bwd(self,i, dy, q, k, v, weights):
        """Actual computation of the gradient of the loss wrt. q, k, and v"""
        d_k = k.shape[-1]
        scale = 1 / np.sqrt(d_k) if self.scale else 1

        dV = weights.swapaxes(-2, -1) @ dy
        dWeights = dy @ v.swapaxes(-2, -1)
        
        #dScores = self.softmax.backward(dWeights)
        grad = self.softQK[i] * dWeights
        sum_grad = np.sum(grad, axis=-1, keepdims=True)
        dScores = grad - self.softQK[i] * sum_grad
        
        dQ = dScores @ k * scale
        dK = dScores.swapaxes(-2, -1) @ q * scale
        return dQ, dK, dV
    
    
    def __call__(self,*args):
        return self.forward(*args)


#test forward/backward dimension
if __name__ == "__main__":
    model = Attention_np(10,20,20,40)
    q = np.random.randn(3,5,10)
    kv = np.random.randn(3,7,20)
    
    output = model(q,kv,kv)
    print(output.shape)
    
    model.backward(output)