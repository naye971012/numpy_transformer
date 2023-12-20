import numpy as np

class Embedding_np:
    """
    embedding layer implemented with numpy
    same functionality to torch.nn.embedding
    """
    
    def __init__(self, num_emb:int , num_dim:int ) -> None:
        """
        Args:
            num_emb (int): the number of the vocab size
            num_dim (int): embedding dimension
        """
        
        #save params and gradient for layer update
        self.params = dict()
        self.grads = dict()
        
        self.num_emb = num_emb
        self.num_dim = num_dim
        
        self.forward_input = None

        self.init_params()
        
    def init_params(self) -> None:
        """
        initalizate params
        in embedding layer, it has 'W' weight tensor
        
        'W' dim = [# of vocab, embedding dimension]
        """
        
        limit = np.sqrt(2 / float(self.num_dim))
        self.params['W'] = np.random.normal(0.0, limit, size=(self.num_emb,self.num_dim))
        
    def forward(self,x:np.array) -> np.array:
        """
        forward process of embedding layer
        
        Args:
            x (np.array[int]): [# of batch, # of vocab(int) ]

        Returns:
            np.array: [# of batch, # of vocab, embedding_dim ]
        """
        
        self.forward_input = x
        output = self.params['W'][x[:]]
        
        return output
        
    def backward(self,d_prev:np.array) -> None:
        """
        backward process of embedding layer
        
        Args:
            d_prev (np.array): [# of batch, # of vocab, embedding_dim]

        Returns:
            None(it should be first layer of the model)
        """
        
        self.grads['dW'] = np.zeros_like(self.params['W'])
        np.add.at(self.grads['dW'], self.forward_input, d_prev)

        return None
    
    def __call__(self,x):
        return self.forward(x)


#check dimension
if __name__=="__main__":
    model = Embedding_np(10,20)
    x = np.random.randint(0,9, size=(1,20))
    output = model(x)
    model.backward(output)