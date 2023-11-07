import numpy as np
from numpy_models.activations.relu import Relu_np
from numpy_models.activations.tanh import Tanh_np

"""
    our rnn is defined as many-to-many type
    
                                      y_1        y_2        y_3
                                       |          |          |
    hidden -> [ w_1 ] -> [ w_2 ] -> [ w_3 ] -> [ w_4 ] -> [ w_5 ]
                 |          |          
                x_1        x_2        


                NOP        NOP        y_1        y_2        y_3
                 |          |          |          |          |
    hidden -> [ w_1 ] -> [ w_2 ] -> [ w_3 ] -> [ w_4 ] -> [ w_5 ]
                 |          |          |          |          |
                x_1        x_2       _zero      _zero      _zero

"""



class RNN_np:
    def __init__(self, input_dim, output_dim ,hidden_dim) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.layers_tanh = None
        self.hidden_list = None
        self.y_preds = None
        
    def forward(self,input_X):
        
        # [# of batch, max length, embedding dim(input_dim) ] -> [# of batch, max length, output_dim]
        
        """
        Computes the forward propagation of the RNN.
        Parameters
        ----------
        input_X : numpy.array or list
            List containing all the inputs that will be used to 
            propagate along the RNN cell.
        Returns
        -------
        y_preds : list
            List containing all the preditions for each input of the
            input_X list.
        """
        self.input_X = input_X
    
        self.layers_tanh = [Tanh_np() for x in input_X]
        hidden = np.zeros((self.hidden_dim , 1))
        
        self.hidden_list = [hidden]
        self.y_preds = []
    
        for input_x, layer_tanh in zip(input_X, self.layers_tanh):
            input_tanh = np.dot(self.Wax, input_x) + np.dot(self.Waa, hidden) + self.b
            hidden = layer_tanh.forward(input_tanh)
            self.hidden_list.append(hidden)
    
            input_softmax = np.dot(self.Wya, hidden) + self.by
            y_pred = self.softmax.forward(input_softmax)
            self.y_preds.append(y_pred)
    
        return self.y_preds
    
    def backward(self,x):
        """
        Computes the backward propagation of the model.
        Defines and updates the gradients of the parameters to used
        in order to actulized the weights.
        """
        gradients = self._define_gradients()
        self.dWax, self.dWaa, self.dWya, self.db, self.dby, dhidden_next = gradients
    
        for index, layer_loss in reversed(list(enumerate(self.layers_loss))):
            dy = layer_loss.backward()
    
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
    
    def __call__(self,x):
        return self.forward(x)