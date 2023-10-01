from typing import Any
import numpy as np

class MaxPooling2D_np:

    def __init__(self, pooling_shape: tuple=(2,2) , stride: int=None) -> None:
        self.pool_w, self.pool_h = pooling_shape
        assert self.pool_h==self.pool_w
        
        self.stride = stride if stride!=None else self.pool_w
        
        self.output_index = None #save where is max

    def forward(self, x:np.array ) -> np.array:
        """

        Args:
            x (np.array): [ # of batch, # of channel, input_height, input_width ]

        Returns:
            np.array: [# of batch, # of channel, output_height, output_width]
        """
        
        n, c, h_in, w_in = x.shape
        h_out = 1 + (h_in - self.pool_h) // self.stride
        w_out = 1 + (w_in - self.pool_w) // self.stride
        self.output = np.zeros((n, c, h_out, w_out))
        self.output_index = np.zeros((n,c,h_in,w_in))

        for i in range(0, h_out):
            for j in range(0,w_out):
                h_start = i * self.stride
                h_end = h_start + self.pool_h
                w_start = j * self.stride
                w_end = w_start + self.pool_w

                sub_array = x[: ,:, h_start:h_end , w_start:w_end]
                self.output[:, :, i, j ] = np.max( sub_array , axis=(2,3) ).reshape(n,c,1,1)
                
                self.output_index[: ,:, h_start:h_end , w_start:w_end] = self.cal_max_idx(sub_array)
                
        return self.output
    
    def cal_max_idx(self, subarray: np.array) -> np.array:
        """
        since numpy argmax cannot use multiple axis, make custom function

        Args:
            subarray (np.array): [n, c, ker_h, ker_w]

        Returns:
            np.array: [n, c, ker_h, ker_w] with binary value (0 or 1)
        """
        n, c, ker_h, ker_w = subarray.shape
        subarray = subarray.reshape(n, c, ker_h *ker_w)
        idx = np.argmax(subarray, axis=2) # [n , c]

        output_array = np.zeros_like(subarray) # [n, c, ker_h * ker_w]
        output_array[np.arange(n)[:, np.newaxis, np.newaxis], np.arange(c)[np.newaxis, :, np.newaxis], idx[:, :, np.newaxis]] = 1
        output_array = output_array.reshape(n,c,ker_h, ker_w)
        return output_array
    
    def backward(self,d_prev:np.array) -> np.array:
        """

        Args:
            d_prev (np.array): [# of batch, # of channel, output_height, output_width]

        Returns:
            np.array: [# of batch, # of channel, input_height, input_width ]
        """
        d_output = np.zeros_like(self.output_index) # [# of batch, # of channel, input_height, input_width]
        n, c , h_out, w_out = d_prev.shape
        
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                h_end = h_start + self.pool_h
                w_start = j * self.stride
                w_end = w_start + self.pool_w
                d_output[: ,:, h_start:h_end , w_start:w_end] = \
                                d_prev[:, :, i:i+1, j:j+1] * self.output_index[: ,:, h_start:h_end , w_start:w_end]
        
        return d_output

    def __call__(self, x:np.array) -> np.array:
        return self.forward(x)


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