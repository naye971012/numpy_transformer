import numpy as np
from numpy_models.activations.relu import Relu_np
from numpy_models.activations.tanh import Tanh_np
from numpy_models.activations.softmax import softmax_np

"""
    our rnn is defined as many-to-many type
    
                           y_1        y_2        y_3
                            |          |          |
    hidden -> [ w_1 ] -> [ w_2 ] -> [ w_3 ] -> [ w_4 ] -> [ w_5 ]
                 |          |          
                x_1        x_2        


                y_1        y_2        y_3       <EOS>      <EOS>
                 |          |          |          |          |
    hidden -> [ w_1 ] -> [ w_2 ] -> [ w_3 ] -> [ w_4 ] -> [ w_5 ]
                 |          |          |          |          |
                x_1        x_2       <EOS>      <EOS>      <EOS>

"""



class RNN_np:
    """
    Recurrent neural network implementation.
    """
    def __init__(self, input_dim, output_dim, hidden_dim):
        """
        Initialize the parameters with the input, output and hidden
        dimensions. 

        Parameters
        ----------
        input_dim : int
            Dimension of the input. 
        output_dim : int
            Dimension of the output.
        hidden_dim : int
            Number of units in the RNN cell.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        params = self._initialize_parameters(
                input_dim, output_dim, hidden_dim
        )
        """
        weights_ya : numpy.array [out_dim , hidden_dim ]
        weights_ax : numpy.array [hidden_dim, input_dim]
        weights_aa : numpy.array [hidden_dim, hidden_dim]
        bias_y : numpy.array [out_dim, 1]
        bias : numpy.array [hidden_dim, 1]
        """
        self.Wya, self.Wax, self.Waa, self.by, self.b = params

    def forward(self, input_X):
        """
        Computes the forward propagation of the RNN.
        Parameters
        ----------
        input_X : numpy.array or list
            List containing all the inputs that will be used to 
            propagete along the RNN cell.
            
        Returns
        -------
        y_preds : np.array
            List containing all the preditions for each input of the
            input_X list.
        """
        #    [ # of batch, max length, embedding dim]
        #--> [max_len, batch, input_size] 으로 바꿔서 사용
        input_X = np.transpose(input_X, (1, 0, 2))
        self.input_X = input_X

        self.layers_tanh = [Tanh_np() for x in input_X]
        self.layers_softmax = [softmax_np() for x in input_X]
        
        hidden = np.zeros((self.hidden_dim , 1))
        self.hidden_list = [hidden]
        
        self.y_preds = []
        
        """
        input: [max_len, batch, input_dim] (20, 1, 300) -> (1,300)
        output: [max_len, batch, output_dim]
        weights_ya : numpy.array [out_dim , hidden_dim ] -> (300, 2)
        weights_ax : numpy.array [hidden_dim, input_dim] (2, 300)
        weights_aa : numpy.array [hidden_dim, hidden_dim] (2, 2)
        bias_y : numpy.array [out_dim, 1]
        bias : numpy.array [hidden_dim, 1] -> (2,1)
        """
        #repeat for max sequence length
        for input_x, layer_tanh, layer_softmax in zip(input_X, self.layers_tanh, self.layers_softmax):
            input_tanh = np.dot(self.Wax, input_x.T) + np.dot(self.Waa, hidden) + self.b
            hidden = layer_tanh.forward(input_tanh)
            self.hidden_list.append(hidden)

            input_softmax = np.dot(self.Wya, hidden) + self.by
            y_pred = layer_softmax.forward(input_softmax) # (300, 1)
            self.y_preds.append(y_pred) #add (output_dim, batch)

        #    [max_len, output_dim, batch] (20, 300, 1)
        #--> [ # of batch, max length, output_dim] 으로 바꿔서 return
        self.y_preds = np.array(self.y_preds)
        self.y_preds = np.transpose(self.y_preds, (2, 0, 1))
        
        return self.y_preds
    

    def backward(self, d_prev):  
        """
        Computes the backward propagation of the model.

        Defines and updates the gradients of the parameters to used
        in order to actulized the weights.
        """
        #    [ # of batch, max length, output dim]
        #--> [max_len, batch, output dim] 으로 바꿔서 사용
        d_prev = np.transpose(d_prev, (1, 0, 2))
        
        gradients = self._define_gradients()
        self.dWax, self.dWaa, self.dWya, self.db, self.dby, dhidden_next = gradients

        for index, cur_d_prev in reversed(list(enumerate(d_prev))):
            dy = cur_d_prev

            # hidden actual
            hidden = self.hidden_list[index + 1]
            hidden_prev = self.hidden_list[index]

            # gradients y
            self.dWya += np.dot(dy, hidden.T)
            self.dby += dy
            dhidden = np.dot(self.Wya.T, dy) + dhidden_next
    
            # gradients a
            dtanh = self.layers_tanh[index].backward(dhidden)
            self.db += dtanh
            self.dWax += np.dot(dtanh, self.input_X[index].T)
            self.dWaa += np.dot(dtanh, hidden_prev.T)
            dhidden_next = np.dot(self.Waa.T, dtanh)


    def clip(self, clip_value):
        """
        Clips the gradients in order to avoisd the problem of 
        exploding gradient.

        Parameters
        ----------
        clip_value : int
            Number that will be used to clip the gradients.
        """
        for gradient in [self.dWax, self.dWaa, self.dWya, self.db, self.dby]:
            np.clip(gradient, -clip_value, clip_value, out=gradient)
        
    def _initialize_parameters(self, input_dim, output_dim, hidden_dim):
        """
        Initialize the parameters randomly.

        Parameters
        ----------
        input_dim : int
            Dimension of the input
        output_dim : int
            Dimension of the ouput
        hidden_dim : int

        Returns
        -------
        weights_y : numpy.array [out_dim , hidden_dim ]
        weights_ax : numpy.array [hidden_dim, input_dim]
        weights_aa : numpy.array [hidden_dim, hidden_dim]
        bias_y : numpy.array [out_dim, 1]
        bias : numpy.array [hidden_dim, 1]
        """
        den = np.sqrt(hidden_dim)

        weights_y = np.random.randn(output_dim, hidden_dim) / den
        bias_y = np.zeros((output_dim, 1))

        weights_ax = np.random.randn(hidden_dim, input_dim) / den
        weights_aa = np.random.randn(hidden_dim, hidden_dim) / den
        bias = np.zeros((hidden_dim, 1))

        return weights_y, weights_ax, weights_aa, bias_y, bias


    def _define_gradients(self):
        """
        Defines the gradients of the model.
        init all grad as zero
        """
        dWax = np.zeros_like(self.Wax)
        dWaa = np.zeros_like(self.Waa)
        dWya = np.zeros_like(self.Wya)

        db = np.zeros_like(self.b)
        dby = np.zeros_like(self.by)

        da_next = np.zeros_like(self.hidden_list[0])

        return dWax, dWaa, dWya, db, dby, da_next
    
    def __call__(self,x):
        return self.forward(x)