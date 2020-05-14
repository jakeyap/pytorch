#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 12:15:14 2020

@author: yongkeong
"""
# This example demonstrates using GPUs for math vs using CPUs

import torch
import time
sizes_to_compare = [200,300,400,500,600,700,800,900]
cpu_times = []
gpu_times = []

size_to_compare = 1000
for size_to_compare in sizes_to_compare:
    ###CPU
    start_time = time.time()
    a = torch.ones(size_to_compare,size_to_compare)
    for _ in range(100000):
        a += a
    elapsed_time = time.time() - start_time
    cpu_times.append(elapsed_time)
    print('CPU time = ',elapsed_time)
    
    ###GPU
    start_time = time.time()
    b = torch.ones(size_to_compare,size_to_compare).cuda()
    for _ in range(100000):
        b += b
    elapsed_time = time.time() - start_time
    gpu_times.append(elapsed_time)
    print('GPU time = ',elapsed_time)
    
import matplotlib.pyplot as plt

figure = plt.figure()
plt.grid(True)
plt.plot(sizes_to_compare, cpu_times, label='CPU')
plt.plot(sizes_to_compare, gpu_times, label='GPU')

plt.legend(loc='best')
plt.ylabel('Time (s)')
plt.xlabel('Array dimension')