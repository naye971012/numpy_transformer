import torch

linear = torch.nn.Linear(5,10)
linear2 = torch.nn.Linear(10,5)

x1 = torch.randn((1,5))
x1.requires_grad =True

x2 = linear(x1)
x3 = linear2(x2)

print(x1)
print(x2)
print(x3)
print("===================")
y = torch.sum(x3)
y.backward()

print(x1.grad)
print(x2.grad)
print(x3.grad)
##
