#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 13:41:50 2020

@author: danbiderman
Dataset classes
"""

import torch
import numpy as np
from arm_models import arm_utils

class Dataset(torch.utils.data.Dataset):
    '''Characterizes a dataset for PyTorch'''
    def __init__(self, data, num_timesteps, max_allowed_chunks):  # list_IDs
        '''Initialization:
        data: np.array of shape (dim_obs, many_timesteps)
        num_timesteps: number of time steps per sequence (chunk)'''
        self.data = data  # cheap to load one 9 by 100K arr.
        self.num_timesteps = int(num_timesteps)
        self.num_valid_chunks = int(
            np.floor(self.data.shape[-1] / self.num_timesteps)) # we won't necessary use them all. dep on max_allowed_chunks
        self.arr_start_IDs = np.arange(0,
                                       num_timesteps * self.num_valid_chunks,
                                       self.num_timesteps)  # not a list
        self.max_allowed_chunks = max_allowed_chunks
        if self.max_allowed_chunks < len(self.arr_start_IDs):
            self.arr_start_IDs = np.random.choice(self.arr_start_IDs,
                                                  size=self.max_allowed_chunks,
                                                 replace = False)
        # take the maximum of allowed chunks from data.

    def __len__(self):
        'Denotes the total number of samples. in our case, each sample is a time series.'
        return len(self.arr_start_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        start_ID = int(self.arr_start_IDs[index])

        # Load data and get label
        data_batch = self.data[:, start_ID:int(start_ID + self.num_timesteps)]
        batch_list = arm_utils.arr_to_list_of_tensors(
            data_batch, 1)  # 1 means that we do not repeat in batch.

        return batch_list, start_ID