#!/usr/bin/env python3
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 17:07:22 2020

@author: jakeyap
"""


"""example_simpleNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Orpn_6WutNzgh8y561Yzhk_C5VxnOG4l

# Creating a basic neural net
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

### BE CAREFUL! PYTORCH NNs ONLY ACCEPT MINIBATCHES. WHEN WORKING ON 
### A SINGLE SAMPLE , USE input.unsqueeze(0) TO ADD A FAKE DIMENSION.

print('Creating a neural net class')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # Similar to MXNet, you just have to define forward function. 
    # Backward propagation is calculated for you.
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

print('Instantiating NN')
net = Net()
print(net)

# Show the weights of the NN. Remember to convert into a list first
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's weight
print(params[1].size(),'\n')  # conv1's bias

print(params[2].size())  # conv2's weight
print(params[3].size(),'\n')  # conv2's bias

print(params[4].size())  # fc1's weight
print(params[5].size(),'\n')  # fc1's bias

print(params[6].size())  # fc2's weight
print(params[7].size(),'\n')  # fc2's bias

print(params[8].size())  # fc3's weight
print(params[9].size(),'\n')  # fc3's bias

print('Feeding an example image')

input_img = torch.randn(1, 1, 32, 32)
print('Input: \n', input_img)
print('Input shape: ', input_img.shape)
out = net(input_img)
print('Output: \n',out)
print('Output shape: ', out.shape)

net.zero_grad()                     # This step zeros the gradients
out.backward(torch.randn(1, 10))    # This step initializes the gradients with random numbers

"""# Define a Loss Function"""

output = net(input_img)
target = torch.randn(10)        # a dummy target label, for example
target = target.view(1, -1)     # make it the same shape as output
criterion = nn.MSELoss()        # MSE means mean-squared-error

loss = criterion(output, target) # calculate the loss
print(loss)

print(loss.backward(retain_graph=True))

print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
loss.grad_fn.next_functions

temp = loss.grad_fn

print(temp.next_functions[0][0].next_functions[0][0])
print(temp.next_functions[0][0].next_functions[1][0])
print(temp.next_functions[0][0].next_functions[2][0])

"""# Backpropagate"""

#To backpropagate the error all we have to do is to loss.backward()
net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

"""# Update Weights"""

import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.02)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input_img)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update

loss

for i in range(10):
    optimizer.zero_grad()   # zero the gradient buffers
    output = net(input_img)
    loss = criterion(output, target)
    print('Loss: ', loss)
    loss.backward()
    optimizer.step()    # Does the update
