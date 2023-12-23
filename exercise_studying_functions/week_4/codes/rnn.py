import numpy as np
from codes.fixed.relu import Relu_np
from codes.fixed.tanh import Tanh_np
from codes.fixed.softmax import softmax_np

"""
    currently, this works only at batch size 1 -> TODO
    our rnn is defined as many-to-many type
    
                y_1        y_2        y_3       <EOS>      <EOS>
                 |          |          |          |          |
    hidden -> [ w_1 ] -> [ w_2 ] -> [ w_3 ] -> [ w_4 ] -> [ w_5 ]
                 |          |          |          |          |
                x_1        x_2       <EOS>      <EOS>      <EOS>

"""


class RNN_np:
    """
    Recurrent neural network implementation with numpy
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
        self.params = dict()
        self.grads = dict()
        
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
        self.params['Wya'] = self.Wya
        self.params['Wax'] = self.Wax
        self.params['Waa'] = self.Waa
        self.params['by'] = self.by
        self.params['b'] = self.b
        
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
            
            ############################# edit here #############################
            input_tanh = None
            hidden = None
            self.hidden_list.append(hidden)

            input_softmax = None
            y_pred = None # (300, 1)
            self.y_preds.append(y_pred) #add (output_dim, batch)

            #####################################################################
            
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
        self.grads['dWya'] = self.dWya
        self.grads['dWax'] = self.dWax
        self.grads['dWaa'] = self.dWaa
        self.grads['dby'] = self.dby
        self.grads['db'] = self.db
        
        """
        input: [max_len, batch, input_dim] (20, 1, 300) -> (1,300)
        output: [max_len, batch, output_dim]
        weights_ya : numpy.array [out_dim , hidden_dim ] -> (300, 2)
        weights_ax : numpy.array [hidden_dim, input_dim] (2, 300)
        weights_aa : numpy.array [hidden_dim, hidden_dim] (2, 2)
        bias_y : numpy.array [out_dim, 1]
        bias : numpy.array [hidden_dim, 1] -> (2,1)
        hidden = [hidden_dim, 1]
        """
        #    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
        #    y -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
        #    c -- The value of the cell state, numpy array of shape (n_a, m, T_x)
        out = list()
        
        for index, cur_d_prev in reversed(list(enumerate(d_prev))):
            self.grads['dy'] = cur_d_prev.T # [batch, input_dim] -> [input_dim, batch]

            # hidden actual
            hidden = self.hidden_list[index + 1]
            hidden_prev = self.hidden_list[index]

            # gradients y
            self.grads['dWya'] += np.dot(self.grads['dy'], hidden.T) #[out_dim,1] * [1, hidden] = [out_dim, hidden]
            self.grads['dby'] += self.grads['dy'] # [out_dim, 1]
            dhidden = np.dot(self.params['Wya'].T, self.grads['dy']) + dhidden_next # [hidden, out] * [out,1] = [hidden, 1]
    
            # gradients a
            dtanh = self.layers_tanh[index].backward(dhidden) # [hidden, 1]
            self.grads['db'] += dtanh # [hidden, 1]
            self.grads['dWax'] += np.dot(dtanh, self.input_X[index]) # [hidden, input]
            self.grads['dWaa'] += np.dot(dtanh, hidden_prev.T) # [hiddne, 1] * [1, hidden] = [hidden, hidden]
            dhidden_next = np.dot(self.params['Waa'].T, dtanh) # [hidden, hidden] * [hidden, 1] = [hidden, 1]

            out.append( self.params['Wax'].T.dot(dtanh) ) # [input, hidden] * [hidden, batch(1)] = [input , batch]
        
        #out =  [max_len, input, batch]
        out = np.array(out)

        #output dim = [batch, max_len, input_dim]
        out = np.transpose(out, (2, 0, 1))
        
        return out

    def clip(self, clip_value):
        """
        Clips the gradients in order to avoisd the problem of 
        exploding gradient.

        Parameters
        ----------
        clip_value : int
            Number that will be used to clip the gradients.
        """
        for gradient in [self.grads['dWax'], self.grads['dWaa'], self.grads['dWya'], self.grads['db'], self.grads['dby']]:
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