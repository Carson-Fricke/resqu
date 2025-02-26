from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import random as r

class SingleVarFunction(Dataset):

  def __init__(self, num_funcs=20, samples=1000, in_range=(-10, 10), function_complexity_range=(1,4), operations=[(1, F.relu), (1, torch.cos), (1, torch.sin), (2, torch.add), (2, torch.mul), (2, torch.sub), (2, torch.pow)], transform=None):
    self.functions = []
    for _ in range(num_funcs):
      pass

  def build_func(self, complexity_range, operations):
    pass
    

  def __len__(self):
    pass

  def __getitem(self, idx):
    pass
