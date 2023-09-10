import numpy as np
import torch

from torch_models.activations.relu import Relu_th
from numpy_models.activations.relu import Relu_np

from torch_models.activations.sigmoid import Sigmoid_th
from numpy_models.activations.sigmoid import Sigmoid_np

def check_correct(prompt, _torch, _numpy, eps=1e-5):
    """
    Print True if two array is same
    
    Args:
        _torch tensor
        _numpy array
    """
    print(prompt, end="  ")
    
    x = _torch.detach().numpy()
    y =_numpy
    
    #check percentage error is less than eps
    if np.allclose(x, y, rtol=eps):
        print("correct!")
    else:
        print("wrong")


def test_activation_functions(prompt , torch_fun, relu_fun):
    print(f"checking {prompt} function ...")
    print("===============================")
    
    ### make random np array or tensor
    numpy_input = np.random.randn(100,100).astype(np.float32) #[100 , 100] shape np array
    torch_input = torch.Tensor(numpy_input).to(torch.float32) #same as numpy input.
    torch_input.requires_grad=True #for autograd utils
    
    ### define activation function
    torch_relu = torch_fun
    numpy_relu = relu_fun
    
    ### calculate function
    torch_output = torch.sum(torch_relu(torch_input)) #torch needs scaler value for backward
    numpy_output = np.sum(numpy_relu(numpy_input),axis=None)
    #print(torch_output , numpy_output)
    
    ### backward
    torch_output.backward() #backward to scaler value. use autograd
    numpy_relu.backward() #compared to torch, numpy needs to call backward function to function itself
    
    ### save gradient
    torch_grad = torch_input.grad
    numpy_grad = numpy_relu.grad
    #print(torch_grad, numpy_grad)
    
    ### check whether implementation is correct
    check_correct("checking forward...", torch_output, numpy_output)
    check_correct("checking backward...", torch_grad , numpy_grad)
    print("===============================\n")

if __name__=="__main__":
    test_activation_functions("Relu" , torch_fun=Relu_th() , relu_fun=Relu_np())
    test_activation_functions("Sigmoid" , torch_fun=Sigmoid_th() , relu_fun=Sigmoid_np())