# -*- coding: utf-8 -*-
"""
Created on Wed May 13 12:15:14 2020

@author: yongkeong
"""


import torch
import time

size_to_compare = 1000

###CPU
start_time = time.time()
a = torch.ones(size_to_compare,size_to_compare)
for _ in range(100000):
    a += a
elapsed_time = time.time() - start_time

print('CPU time = ',elapsed_time)

###GPU
start_time = time.time()
b = torch.ones(size_to_compare,size_to_compare).cuda()
for _ in range(100000):
    b += b
elapsed_time = time.time() - start_time

print('GPU time = ',elapsed_time)