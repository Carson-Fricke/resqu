import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from src.nn_modules.layers import SplitReSqULinear

from ..nn_modules.trainingmodel import TrainingModel

from math import floor, ceil



class SingleVarRegresser(TrainingModel):
  def __init__(self, width:int, depth:int, resqu_rate: float=0):
    self.width = width
    self.depth = depth
    super(SingleVarRegresser, self).__init__()
    relu_rate = 1 - resqu_rate
    self.in_layer = SplitReSqULinear(1, floor(relu_rate*width), ceil(resqu_rate*width), norm=False, bound_resqu=False)
    tml = depth * [SplitReSqULinear(width, floor(relu_rate*width), ceil(resqu_rate*width), norm=False, bound_resqu=False)]
    self.middle = nn.Sequential(*tml)
    self.out_layer = nn.Linear(width, 1)
  
  def forward(self, x):
    # if (x.shape[-1] != 1) :
    #   x = x.unsqueeze(-1)
    
    out = self.in_layer(x)
    out = self.middle(out)
    out = self.out_layer(out)

    # if (out.shape[-1] != 1):
    #   out = out.unsqueeze(-1)

    return out
  
  def resqu_wb_norm(self):
    children = [child for child in list(self.middle) if isinstance(child, SplitReSqULinear)] + [self.in_layer]
    wb = zip(*[child.resqu_wb_norm() for child in children])

    return tuple(list(item) for item in wb)