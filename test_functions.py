import numpy as np
import torch

from torch_models.activations.relu import Relu_th
from numpy_models.activations.relu import Relu_np

from torch_models.activations.sigmoid import Sigmoid_th
from numpy_models.activations.sigmoid import Sigmoid_np

from torch_models.losses.binary_ce import Binary_Cross_Entropy_th
from numpy_models.losses.binary_ce import Binary_Cross_Entropy_np

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


def test_activation_functions(prompt , torch_fun, numpy_fun):
    print(f"checking {prompt} activation function ...")
    print("===============================")
    
    ### make random np array or tensor
    numpy_input = np.random.randn(100,100).astype(np.float32) #[100 , 100] shape np array
    torch_input = torch.Tensor(numpy_input).to(torch.float32) #same as numpy input.
    torch_input.requires_grad=True #for autograd utils
    
    ### define activation function
    torch_activation_fun = torch_fun
    numpy_activation_fun = numpy_fun
    
    ### calculate function
    torch_output = torch.sum(torch_activation_fun(torch_input)) #torch needs scaler value for backward
    numpy_output = np.sum(numpy_activation_fun(numpy_input),axis=None)
    #print(torch_output , numpy_output)
    
    ### backward
    torch_output.backward() #backward to scaler value. use autograd
    numpy_activation_fun.backward() #compared to torch, numpy needs to call backward function to function itself
    
    ### save gradient
    torch_grad = torch_input.grad
    numpy_grad = numpy_activation_fun.grad
    #print(torch_grad, numpy_grad)
    
    ### check whether implementation is correct
    check_correct("checking forward...", torch_output, numpy_output)
    check_correct("checking backward...", torch_grad , numpy_grad)
    print("===============================\n")


def test_loss_functions(prompt , torch_fun, numpy_fun, is_prob=False):
    print(f"checking {prompt} loss function ...")
    print("===============================")
    
    ### make random np array or tensor
    if is_prob:
        numpy_input = np.random.rand(1,100).astype(np.float32) #[1 , 100] shape np array, prob
        numpy_target = np.random.randint(0,2, size=(1,100)).astype(np.float32) #[1,100] shape with binary value
    else:
        numpy_input = np.random.randn(1,100).astype(np.float32) #[1 , 100] shape np array, logit
        numpy_target = np.random.randn(1,100).astype(np.float32) #[1, 100] shape with mean=0, std=1 value

    torch_input = torch.Tensor(numpy_input).to(torch.float32) #same as numpy input.
    torch_target = torch.Tensor(numpy_target).to(torch.float32) #same as numpy target
    
    torch_input.requires_grad=True #for autograd utils
    
    ### define activation function
    torch_loss_fun = torch_fun
    numpy_loss_fun = numpy_fun
    
    ### calculate function
    torch_output = torch_loss_fun(torch_input, torch_target)

    numpy_output = numpy_loss_fun(numpy_input, numpy_target)
    #print(torch_output , numpy_output)
    
    ### backward
    torch_output.backward() #backward to scaler value. use autograd
    numpy_loss_fun.backward() #compared to torch, numpy needs to call backward function to function itself
    
    ### save gradient
    torch_grad = torch_input.grad
    numpy_grad = numpy_loss_fun.grad
    #print(torch_grad, numpy_grad)
    
    ### check whether implementation is correct
    check_correct("checking forward...", torch_output, numpy_output)
    check_correct("checking backward...", torch_grad , numpy_grad)
    print("===============================\n")

if __name__=="__main__":
    test_activation_functions("Relu" , torch_fun=Relu_th() , numpy_fun=Relu_np())
    test_activation_functions("Sigmoid" , torch_fun=Sigmoid_th() , numpy_fun=Sigmoid_np())
    
    test_loss_functions("Binary Cross Entropy", torch_fun=Binary_Cross_Entropy_th(), numpy_fun=Binary_Cross_Entropy_np() , is_prob=True )