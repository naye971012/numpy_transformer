import numpy as np

def positional_encoding(max_position:int, d_model:int, min_freq:int=1e-4) -> np.array:
        
    position = np.arange(max_position) # [max position]
    freqs = min_freq**(2*(np.arange(d_model)//2)/d_model) # [d_model]
    pos_enc = position.reshape(-1,1)*freqs.reshape(1,-1) #[max_position , 1] * [1, d_model] = [ max position , d_model ]
    pos_enc[:, ::2] = np.cos(pos_enc[:, ::2])
    pos_enc[:, 1::2] = np.sin(pos_enc[:, 1::2])
  
    return pos_enc

class Embedding_with_positional_encoding_np:
    def __init__(self, num_emb:int, num_dim:int) -> None:
        self.params = dict()
        self.grads = dict()
        
        self.num_emb = num_emb
        self.num_dim = num_dim
        
        self.forward_input = None
        self.pos_enc = positional_encoding(num_emb,num_dim)
        
        limit = np.sqrt(2 / float(num_dim))
        self.params['W'] = np.random.normal(0.0, limit, size=(num_emb,num_dim))

    def forward(self,x:np.array) -> np.array:
        """

        Args:
            x (np.array[int]): [# of batch, # of vocab(int) ]

        Returns:
            np.array: [# of batch, # of vocab, embedding_dim ]
        """
        
        self.forward_input = x
        output = self.params['W'][x[:]] + self.pos_enc[ :x.shape[1]]
        
        return output
        
    def backward(self,d_prev:np.array) -> np.array:
        """

        self.W = [# of embedding , embedding_dim]
        
        Args:
            d_prev (np.array): [# of batch, # of vocab, embedding_dim]

        Returns:
            np.array: _description_
        """
        
        self.grads['dW'] = np.zeros_like(self.params['W'])
        np.add.at(self.grads['dW'], self.forward_input, d_prev)
        
        """
        b, vocab, dim = d_prev.shape
        vocab_len, dim = self.params['W'].shape
        
        expanded_d_prev = np.zeros(shape=(b,vocab_len,dim))
        expanded_d_prev[:,self.forward_input[:]] = d_prev
        
        self.grads['dW'] = np.mean(expanded_d_prev,axis=0)
        """
        
        return None
    
    def __call__(self,x):
        return self.forward(x)

if __name__=="__main__":
    
    model = Embedding_with_positional_encoding_np(10,20)
    x = np.random.randint(0,9, size=(1,5))
    output = model(x)
    model.backward(output)

    """
    import matplotlib.pyplot as plt
    ### Plotting ####
    d_model = 128
    max_pos = 256
    mat = positional_encoding(max_pos, d_model)
    plt.pcolormesh(mat, cmap='copper')
    plt.xlabel('Depth')
    plt.xlim((0, d_model))
    plt.ylabel('Position')
    plt.title("PE matrix heat map")
    plt.colorbar()
    plt.show()
    """