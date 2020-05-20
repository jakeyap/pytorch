#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 11:13:04 2020

@author: jakeyap
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook
import time


prefix = 'data/yelp/'

time_start = time.time()
print("Print some training data")
train_df = pd.read_csv(prefix + 'train.csv', header=None)
print(train_df.head())

print("\nPrint some test data")
test_df = pd.read_csv(prefix + 'test.csv', header=None)
print(test_df.head())

print("\nThe original labels are 1s and 2s")
print("\nReformating the labels to 1s and 0s")
train_df[0] = (train_df[0] == 2).astype(int)
test_df[0] = (test_df[0] == 2).astype(int)
train_df = pd.read_csv(prefix + 'train.csv', header=None)
print(train_df.head())



print("\nFormat data for transformers library")
print("ID | label |  row id")
print
dev_df = pd.DataFrame({
    'id':range(len(test_df)),
    'label':test_df[0],
    'alpha':['a']*test_df.shape[0],
    'text': test_df[1].replace(r'\n', ' ', regex=True)
})

dev_df.head()

train_df.to_csv('data/yelp/train.tsv', sep='\t', index=False, header=False)
dev_df.to_csv('data/yelp/dev.tsv', sep='\t', index=False, header=False)

''' DO NOT TOUCH BELOW''' 

time_end = time.time()
time_elapsed = time_end - time_start

print("\nTime taken: %6.2f s" % time_elapsed)