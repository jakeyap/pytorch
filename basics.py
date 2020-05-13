# -*- coding: utf-8 -*-
"""
Created on Tue May  5 18:02:01 2020

@author: Yong Keong
"""
import numpy as np
import torch

print('------ this creates an empty matrix, with garbage in RAM ------')
x = torch.empty(5, 3)
print(x,'\n')

print('------ this creates a matrix, initialized randomly ------')
x = torch.rand(5, 3)
print(x,'\n')

print('------ this creates a 0 matrix, long data type ------')
x = torch.zeros(5, 3, dtype=torch.long)
print(x,'\n')

print('------ this creates a tensor directly from data ------')
x = torch.tensor([5.5, 3])
print(x,'\n')

print('------ this shows conversion to & from a numpy array ------')
a = np.ones(5)
b = torch.from_numpy(a)
c = b.numpy()

np.add(a, 1, out=a)
print('a: ', a)
print('b: ', b)
print('c: ', c)

print('let us run this cell only if CUDA is available')
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    print('We will use "torch.device" objects to move tensors in and out of GPU','\n')
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!
else:
    print("cuda not available\n")
    

print("------ Track computations on this number ------")
print("------ Set the requires grad flag to true------")
x = torch.ones(2, 2, requires_grad=True)
print(x,'\n')
print("------ Do a tensor operation, add 2 to x ------")
y = x + 2
print("------ There is now a gradient function ------")
print(y,'\n')

print("------ Compute z = y x y x 3 ------")
z = y * y * 3
print(z,'\n')

print("------ print out = mean(z)------")
out = z.mean()
print(out,'\n')
print("------ Calculate the gradient of out wrt x ------")
out.backward()
print(x.grad)

