#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 14:43:04 2020

@author: jakeyap
"""

import logging
import os
import random
import json

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

import transformers

from transformers import (WEIGHTS_NAME, BertConfig, 
                          BertForSequenceClassification, BertTokenizer)


from example_yelp_utils import (convert_examples_to_features,
                        output_modes, processors)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

'''
from transformers import tokenization_bert
from transformers import BertModel, BertConfig

model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

#tokenizer = transformers.
#transformers.BertTokenizer
'''

'''
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
sentence1 = 'hi there you idiot.'
sentence2 = 'i was here since twenty hours ago.'
tokenized_sentence1 = tokenizer.tokenize(text=sentence1)
tokenized_sentence2 = tokenizer.tokenize(text=sentence2)
encoded_sentence = tokenizer.encode(text=tokenized_sentence1,
                                    text_pair=tokenized_sentence2,
                                        max_length=11, 
                                        pad_to_max_length=True)

print(tokenizer.decode(encoded_sentence))
'''