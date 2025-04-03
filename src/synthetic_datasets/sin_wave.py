import copy
import functools
import re
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import random as r
import matplotlib.pyplot as plt

class SingleVarFunctions(Dataset):
  def __init__(self, input_range=(-4,4), step=0.01, noise=0.5, seed=777):
    self.x = torch.arange(start=input_range[0], end=input_range[1], step=step)
    self.y = torch.sin(self.x) + noise * torch.rand_like(self.x)


  def __len__(self):
    return len(self.items)

  @functools.cache
  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]
