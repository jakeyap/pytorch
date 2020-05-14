#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:09:15 2020

@author: jakeyap
"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

''' ======== SETTING HYPERPARAMS ======== '''
N_EPOCHS = 5
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 1000
LEARNING_RATE = 0.01
MOMENTUM = 0.5
LOG_INTERVAL = 10

PRINT_PICTURE = False

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

from pathlib import Path
import requests

# Default GPU
cuda = torch.device('cuda')

''' ======== IMPORTING THE DATA ======== '''
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

print('IMPORTING MNIST DATA')
if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

''' ======== EXTRACTING THE DATA ======== '''
import pickle
import gzip

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), (x_tests, y_tests)) = pickle.load(f, encoding="latin-1")

''' ======== DISPLAY 3 DATAPOINTS ======== '''
if (PRINT_PICTURE):
    # data is formatted as 50k rows x 784 cols
    # each col is one training sample that is actually 28x28 pixels
    img0 = x_train[0].reshape((28, 28))
    img1 = x_train[1].reshape((28, 28))
    img2 = x_train[2].reshape((28, 28))
    
    combined_img = np.zeros(shape=(28,28*3))
    combined_img[:,28*0:28*1] = img0
    combined_img[:,28*1:28*2] = img1
    combined_img[:,28*2:28*3] = img2
    plt.imshow(combined_img, cmap="gray")
    print(type(x_train))

''' ======== CHANGE TO TORCH DATA ======== '''
x_train, y_train, x_valid, y_valid, x_tests, y_tests = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid, x_tests, y_tests)
)

x_train = x_train.to(cuda)
y_train = y_train.to(cuda)
x_valid = x_valid.to(cuda)
y_valid = y_valid.to(cuda)
x_tests = x_tests.to(cuda)
y_tests = y_tests.to(cuda)

n, c = x_train.shape
x_train, x_train.shape, y_train.min(), y_train.max()
print("x_train shape: ",x_train.shape)
print("y_train shape: ",y_train.shape)

size_train = x_train.shape[0]
size_valid = x_valid.shape[0]
size_tests = x_tests.shape[0]

x_train = x_train.reshape(size_train,1,28,28)
x_valid = x_valid.reshape(size_valid,1,28,28)
x_tests = x_tests.reshape(size_tests,1,28,28)

''' ======== CONVERT TO TENSORDATASET ======== '''
''' ======== PLUG INTO A DATALOADER ======== '''

from torch.utils.data import DataLoader, TensorDataset
train_dataset = TensorDataset(x_train,y_train)
valid_dataset = TensorDataset(x_valid,y_valid)
tests_dataset = TensorDataset(x_tests,y_tests)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE_TEST)
tests_loader = DataLoader(tests_dataset, batch_size=BATCH_SIZE_TEST)

examples = enumerate(tests_loader)
batch_idx, (example_data, example_targets) = next(examples)
print("\none minibatch's shape is: ", example_data.shape)
print("Format: <minibatch size>, <num of channels>, <rows>, <cols>\n")


''' ======== CREATE A CNN MODEL ======== '''
class myCNN(nn.Module):
    def __init__(self):
        super(myCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

# Instantiate model
network = myCNN()
# Move onto a GPU
network.to(cuda)
# Define loss function as negative likelihood loss
loss_function = F.nll_loss
# Use SGD algo for training
optimizer = optim.SGD(network.parameters(), lr=LEARNING_RATE,
                      momentum=MOMENTUM)

# Variables to store losses
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(N_EPOCHS + 1)]

# Training algorithm
# 
def train(epoch):
    # This function trains 1 epoch
    
    # Set network into training mode
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Reset gradients to prevent accumulation
        optimizer.zero_grad()
        # Forward prop
        output = network(data)
        # Calculate loss
        loss = loss_function(output, target)
        # Backward prop find gradients
        loss.backward()
        # Update weights & biases
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, batch_idx * len(data), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader), loss.item()))
            
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset))
            )
            
            # Store the states of model and optimizer into logfiles
            # In case training gets interrupted, you can load old states
            torch.save(network.state_dict(), './results/model_state.pth')
            torch.save(optimizer.state_dict(), './results/optimizer.pth')
            
def test():
    # This function evaluates the entire test set
    
    # Set network into evaluation mode
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batchid, (data, target) in enumerate(tests_loader):
            output = network(data)
            test_loss += loss_function(output, target, size_average=False).item()
            #test_loss += loss_function(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(tests_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
          test_loss, correct, len(tests_loader.dataset),
          100. * correct / len(tests_loader.dataset)))
    
test()
for epoch in range(1, N_EPOCHS + 1):
    train(epoch)
    test()
    
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
fig