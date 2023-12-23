from typing import Any
import numpy as np

class MaxPooling2D_np:
    """
    maxpooling layer 2D implemented with numpy
    same functionality as nn.Maxpool2D
    """
    
    def __init__(self, pooling_shape: tuple=(2,2) , stride: int=None) -> None:
        """
        Args:
            pooling_shape (tuple):  Defines the shape of the window used in the pooling operation. Defaults to (2,2).
            stride (int):  Indicates the stride value, determining the movement of the window during the pooling operation. Defaults to None.
        """
        #width and height should be same
        self.pool_w, self.pool_h = pooling_shape
        assert self.pool_h==self.pool_w
        
        #check whether stride is defined
        self.stride = stride if stride!=None else self.pool_w
        
        self.output_index = None #save where is max

    def forward(self, x:np.array ) -> np.array:
        """
        forward process of maxpooling2D layer
        
        Args:
            x (np.array): [ # of batch, # of channel, input_height, input_width ]

        Returns:
            np.array: [# of batch, # of channel, output_height, output_width]
        """
        
        #check input dimension
        n, c, h_in, w_in = x.shape
        
        #check output dimension
        h_out = 1 + (h_in - self.pool_h) // self.stride
        w_out = 1 + (w_in - self.pool_w) // self.stride
        
        #save forward output
        self.output = np.zeros((n, c, h_out, w_out))
        
        #save index of max for backward process
        self.output_index = np.zeros((n,c,h_in,w_in))

        #sliding window of (pool_h,pool_w) kernel
        for i in range(0, h_out):
            for j in range(0,w_out):
                
                #calculate start/end value
                h_start = i * self.stride
                h_end = h_start + self.pool_h
                w_start = j * self.stride
                w_end = w_start + self.pool_w

                #space where current kernel locate
                sub_array = x[: ,:, h_start:h_end , w_start:w_end]
                
                #save forward output & max index
                self.output[:, :, i, j ] = np.max( sub_array , axis=(2,3) ).reshape(n,c,1,1)
                self.output_index[: ,:, h_start:h_end , w_start:w_end] = self.cal_max_idx(sub_array)
                
        return self.output
    
    def cal_max_idx(self, subarray: np.array) -> np.array:
        """
        since numpy argmax cannot use multiple axis, make custom function for calculate max idx

        Args:
            subarray (np.array): [n, c, ker_h, ker_w]

        Returns:
            np.array: [n, c, ker_h, ker_w] with binary value (0 or 1)
        """
        
        n, c, ker_h, ker_w = subarray.shape
        
        #make subarray flatten and calculate max index
        subarray = subarray.reshape(n, c, ker_h *ker_w)
        idx = np.argmax(subarray, axis=2) # [n , c]

        #reshape flatten subarray into original shape
        output_array = np.zeros_like(subarray) # [n, c, ker_h * ker_w]
        output_array[np.arange(n)[:, np.newaxis, np.newaxis], np.arange(c)[np.newaxis, :, np.newaxis], idx[:, :, np.newaxis]] = 1
        output_array = output_array.reshape(n,c,ker_h, ker_w)
        return output_array
    
    def backward(self,d_prev:np.array) -> np.array:
        """
        backward process of maxpooling2D layer
        
        Args:
            d_prev (np.array): [# of batch, # of channel, output_height, output_width]

        Returns:
            np.array: [# of batch, # of channel, input_height, input_width ]
        """
        
        d_output = np.zeros_like(self.output_index) # [# of batch, # of channel, input_height, input_width]
        n, c , h_out, w_out = d_prev.shape
        
        #moving subarray
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + self.pool_h
                w_start = j * self.stride
                w_end = w_start + self.pool_w
                
                ############# edit here #############
                
                d_output[: ,:, h_start:h_end , w_start:w_end] = None
                
                #####################################
                
        return d_output

    def __call__(self, x:np.array) -> np.array:
        return self.forward(x)


#check dimension
if __name__== "__main__":
    model = MaxPooling2D_np()
    x = np.random.randint(1,10,size=(1,1,4,4))
    print(x)
    print("==============")
    output = model(x)
    print(output)
    print("==============")
    print(model.output_index)
    print("==============")
    print(model.backward(output))