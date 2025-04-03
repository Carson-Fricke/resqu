import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nn_modules.layers import SplitReSqULinear

from ..nn_modules.trainingmodel import TrainingModel

from math import floor, ceil



class MnistAutoEncoder(TrainingModel):
  def __init__(self, resqu_rate=0):
    super(MnistAutoEncoder, self).__init__()
    relu_rate = 1 - resqu_rate
    self.encoder = nn.Sequential(
      SplitReSqULinear(784, ceil(relu_rate*250), floor(resqu_rate*250)),
      SplitReSqULinear(250, ceil(relu_rate*100), floor(resqu_rate*100)),
      SplitReSqULinear(100, ceil(relu_rate*15), floor(resqu_rate*15))
    )
    self.decoder = nn.Sequential(
      SplitReSqULinear(15, ceil(relu_rate*100), floor(resqu_rate*100)),
      SplitReSqULinear(100, ceil(relu_rate*250), floor(resqu_rate*250)),
      SplitReSqULinear(250, ceil(relu_rate*784), floor(resqu_rate*784))
    )
  
  def forward(self, x):
    out = x.view(-1, 784).squeeze()
    out = self.encoder(out)
    out = self.decoder(out)
    return out.view(-1, 28, 28)
