from __future__ import division
from torch.utils.data import Dataset
import numpy as np
import sys
import random
import copy
import torch
from sklearn import preprocessing
# np.random.seed(1)
 
class DataHelper(Dataset):
    def __init__(self, node_list, transform=None):
        self.num_nodes = len(node_list)
        self.transform = transform
        self.node_list = node_list
 
    def __len__(self):
        return self.num_nodes
 
    def __getitem__(self, idx):
 
        # node_idx = np.arange(self.num_nodes)[idx]
        node_idx = self.node_list[idx]
 
        sample = {
            'node_idx': node_idx,
        }
 
        if self.transform:
            sample = self.transform(sample)
 
        return sample
