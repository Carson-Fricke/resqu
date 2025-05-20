import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from src.nn_modules.layers import SplitReSqULinear

from ..nn_modules.trainingmodel import TrainingModel

from math import floor, ceil



class LinearAutoEncoder(TrainingModel):
  def __init__(self, dims, channels, resqu_rate=0):
    self.dims = dims
    self.channels = channels
    self.size_in = reduce(lambda x,y: x*y, dims) * channels
    super(LinearAutoEncoder, self).__init__()
    relu_rate = 1 - resqu_rate
    self.encoder = SplitReSqULinear(self.size_in, ceil(relu_rate*36), floor(resqu_rate*36), secondary_bias=True, bound_resqu=False, norm=False)
    
    self.decoder = nn.Linear(36, self.size_in)
    
  
  def forward(self, x):
    
    out = x.view(-1, self.size_in).squeeze()
    out = self.encoder(out)

    out = self.decoder(out)
    return F.sigmoid(out.view(-1, self.channels, *self.dims).squeeze())
  
  def resqu_wb_norm(self):
    return self.encoder.resqu_wb_norm()