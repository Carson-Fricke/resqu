from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nn_modules.layers import SplitReSqULinear, SplitReSqUConv, SplitReSqUConvTranspose


from ..nn_modules.trainingmodel import TrainingModel

from math import floor, ceil



class ConvAutoEncoder(TrainingModel):
  def __init__(self, dims, channels, resqu_rate=0):
    self.dims = dims
    self.channels = channels
    self.size_in = reduce(lambda x,y: x*y, dims) * channels
    super(ConvAutoEncoder, self).__init__()
    relu_rate = 1 - resqu_rate
    self.encoder = nn.Sequential(
      SplitReSqUConv(channels, floor(16 * relu_rate), ceil(16 * resqu_rate), kernel_size=3, stride=1, padding=1),
      nn.AvgPool2d(kernel_size=2, stride=2),
      SplitReSqUConv(16, floor(8 * relu_rate), ceil(8 * resqu_rate), kernel_size=3, stride=1, padding=1),
      nn.AvgPool2d(kernel_size=2, stride=2),
      SplitReSqUConv(8, floor(4 * relu_rate), ceil(4 * resqu_rate), kernel_size=2, stride=1, padding=0),
      nn.AvgPool2d(kernel_size=2, stride=2)
    )
    self.decoder = nn.Sequential(
      SplitReSqUConvTranspose(4, floor(8 * relu_rate), ceil(8 * resqu_rate), 
                         kernel_size=4, 
                         stride=2, 
                         padding=1, 
                         output_padding=1),
      SplitReSqUConvTranspose(8, floor(16 * relu_rate), ceil(16 * resqu_rate), 
                         kernel_size=3, 
                         stride=2, 
                         padding=1, 
                         output_padding=1),
      nn.ConvTranspose2d(16, channels, 
                         kernel_size=3, 
                         stride=2, 
                         padding=1, 
                         output_padding=1),
      nn.Sigmoid()
    )
  
  def forward(self, x):
    if (x.dim() < 4 and x.size(0) != 1):
      x = x.unsqueeze(1)
    out = self.encoder(x)
    out = self.decoder(out)
    return out.view(-1, 28, 28)
  
  def resqu_wb_norm(self):
    children = [child for child in list(self.encoder) if isinstance(child, SplitReSqUConv) or isinstance(child, SplitReSqUConvTranspose)] + [child for child in list(self.decoder) if isinstance(child, SplitReSqUConv) or isinstance(child, SplitReSqUConvTranspose)]
    wb = zip(*[child.resqu_wb_norm() for child in children])
    return tuple(list(item) for item in wb)
