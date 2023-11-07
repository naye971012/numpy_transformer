import numpy as np

class Embedding_np:
    def __init__(self, num_emb, num_dim) -> None:
        self.num_emb = num_emb
        self.num_dim = num_dim
        
        self.forward_input = None
        
        limit = np.sqrt(2 / float(num_dim))
        self.W = np.random.normal(0.0, limit, size=(num_emb,num_dim))
        
    def forward(self,x:np.array) -> np.array:
        """

        Args:
            x (np.array[int]): [# of batch, # of vocab(int) ]

        Returns:
            np.array: [# of batch, # of vocab, embedding_dim ]
        """
        
        self.forward_input = x
        output = self.W[x[:]]
        
        return output
        
    def backward(self,d_prev:np.array) -> np.array:
        """

        self.W = [# of embedding , embedding_dim]
        
        Args:
            d_prev (np.array): [# of batch, # of vocab, embedding_dim]

        Returns:
            np.array: _description_
        """
        
        #b, vocab, dim = d_prev.shape
        #vocab_len, dim = self.W.shape
        
        self.dW = np.zeros_like(self.W)
        np.add.at(self.dW, self.forward_input, d_prev)
        
        #expanded_d_prev = np.zeros(shape=(b,vocab_len,dim))
        #expanded_d_prev[:,self.forward_input[:]] = d_prev
        #self.dW = np.mean(expanded_d_prev,axis=0)

        return None
    
    def __call__(self,x):
        return self.forward(x)

if __name__=="__main__":
    model = Embedding_np(10,20)
    x = np.random.randint(0,9, size=(1,5))
    output = model(x)
    model.backward(output)