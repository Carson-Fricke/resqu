import copy
import functools
import re
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import random as r
import matplotlib.pyplot as plt

class SineWave(Dataset):
  def __init__(self, input_range=(-4,4), step=0.01, noise=0.2, seed=777):
    self.x = torch.arange(start=input_range[0], end=input_range[1], step=step)
    self.y = torch.sin(self.x) + noise * torch.rand_like(self.x)


  def __len__(self):
    return len(self.x)

  @functools.cache
  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]
  
class Polynomial(Dataset):
  def __init__(self, input_range=(-2,2), step=0.01, noise=0.2, seed=777):
    self.x = torch.arange(start=input_range[0], end=input_range[1], step=step)
    self.y = torch.pow(self.x, 3) - 2 * self.x + noise * torch.rand_like(self.x)


  def __len__(self):
    return len(self.x)

  @functools.cache
  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]
