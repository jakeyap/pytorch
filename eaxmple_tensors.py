#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:48:58 2020

@author: Yong Keong
"""

import torch

### Generate some data
torch.manual_seed(7) # Set the random seed so things are predictable

# Features are 5 random normal variables
features = torch.randn((1, 5))
# True weights for our data, random normal variables again
weights = torch.randn_like(features)
# and a true bias term
bias = torch.randn((1, 1))


# Now, make our labels from our data and true weights with 2 ways
y1 = torch.sigmoid(torch.sum(features * weights) + bias)
y2 = torch.sigmoid((features * weights).sum() + bias)

print(y1)
print(y2)

### Generate some data
torch.manual_seed(7) # Set the random seed so things are predictable

# Features are 3 random normal variables
features = torch.randn((1, 3))

# Define the size of each layer in our network
n_input = features.shape[1]     # Number of input units, must match number of input features
n_hidden = 2                    # Number of hidden units 
n_output = 1                    # Number of output units

# Weights for inputs to hidden layer
W1 = torch.randn(n_input, n_hidden)
# Weights for hidden layer to output layer
W2 = torch.randn(n_hidden, n_output)

# and bias terms for hidden and output layers
B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))

h = torch.sigmoid(torch.mm(features, W1) + B1)
output = torch.sigmoid(torch.mm(h, W2) + B2)
print(output)

### conversion to and from numpy
import numpy as np
a = np.random.rand(4,3)
print(a)