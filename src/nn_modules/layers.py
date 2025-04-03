import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import math
from typing import Tuple
from .activations import ReSqU


# class MultiActivationDense(nn.Module):
  
#   activation_pairs: list[Tuple[nn.Module, int]]
#   out_size: int
#   in_size: int
#   w: nn.Parameter
#   b: nn.Parameter
#   def __init__(self, in_size, activations: list[Tuple[nn.Module, int] | Tuple[nn.Module, int, bool]]):
#     super().__init__()
#     self.activation_pairs = activations
#     self.in_size, self.out_size = in_size, sum(n for _, n in self.activation_pairs)
#     weights = torch.Tensor(self.out_size, self.in_size)
#     self.w = nn.Parameter(weights)
#     biases = torch.Tensor(self.out_size)
#     self.b = nn.Parameter(biases)

#     # parameter initialization
#     nn.init.kaiming_uniform_(self.w)
#     fi , _ = nn.init._calculate_fan_in_and_fan_out(self.w)
#     br = 1 / math.sqrt(fi)
#     nn.init.uniform_(self.b, -br, br)
#     # print(self.out_size)

#   def forward(self, x):
#     z = torch.add(torch.mm(x, self.w.t()), self.b)
#     zs = torch.tensor_split(z, tuple(map(lambda x: x[1], self.activation_pairs))[:-1])
#     zsa = map(lambda x: x[1][0](x[0]), zip(zs, self.activation_pairs))
#     a = torch.cat(tuple(zsa), 0)
#     return a


# class MultiActivationConv(nn.Module):

#   activation_pairs: list[Tuple[nn.Module, int]]
#   out_channels: int
#   in_channels: int
#   conv: nn.Conv2d
  
#   def __init__(self, in_channels, activations: list[Tuple[nn.Module, int]], kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, norm=True):
#     super().__init__()
#     self.activation_pairs = activations
#     self.in_channels, self.out_channels = in_channels, sum(n for _, n in self.activation_pairs)
#     self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
#     # self.norm = nn.BatchNorm2d(self.out_channels)

#   def forward(self, x):
#     z = self.conv(x)
#     zs = torch.tensor_split(z, tuple(map(lambda x: x[1], self.activation_pairs))[:-1])
#     zsa = map(lambda x: x[1][0](x[0]), zip(zs, self.activation_pairs))
#     a = torch.cat(tuple(zsa), 0)
#     return a


# class VarianceSplitConv(nn.Module):
   
#   def __init__(self, in_channels, activations: list[Tuple[nn.Module, int]], kernel_size, var_channels=1, stride=1, padding=0, dilation=1, groups=1, bias=True, norm=False):
#     super().__init__()
#     self.activation_pairs = activations
#     self.in_channels, self.out_channels = in_channels, sum(n for _, n in self.activation_pairs)
#     self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
#     self.vconv = nn.Conv2d(self.in_channels, var_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=False)
#     # self.norm = nn.InstanceNorm2d(self.out_channels) if norm else nn.Identity()
#     # nn.init.xavier_uniform_(self.conv.weight, 0.01)
  
#   def forward(self, x):
#     z = self.norm(self.conv(x))
#     zs = torch.tensor_split(z, tuple(map(lambda x: x[1], self.activation_pairs))[:-1])
#     zsa = map(lambda x: x[1][0](x[0]), zip(zs, self.activation_pairs))
#     x2 = torch.pow(x, 2)
#     v = F.relu(self.vconv(x2))
    
#     a = torch.cat(tuple(zsa), dim=0)
#     a = torch.cat((a, v), dim=1)
#     return a




# class InteractionConv(nn.Module):
#   def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
#     super(InteractionConv, self).__init__()
#     self.in_channels, self.out_channels = in_channels, out_channels
#     self.convr = nn.Conv2d(self.in_channels, self.out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
#     #self.norm = nn.InstanceNorm2d(self.out_channels) if norm else nn.Identity()
#     nn.init.xavier_uniform_(self.convr.weight, 0.1)

#   def forward(self, x):
#     z = torch.pow(self.conv(x), 2)
#     x2 = torch.pow(x, 2)
#     v = F.conv2d(x2, torch.pow(self.convr.weight, 2), self.convr.bias, self.convr.stride, padding=self.convr.padding, dilation=self.convr.dilation, groups=self.convr.groups)
#     return F.relu(z-v)

class SplitReSqULinear(nn.Module):
  def __init__(self, in_features, relu_features, resqu_features, bias=True, resqu_bias=True, norm=True, bound_resqu=True):
    super(SplitReSqULinear, self).__init__()
    self.relu_linear = nn.Linear(in_features, relu_features, bias=bias)
    self.resqu_linear = None
    if resqu_features != 0:
      self.resqu_linear = nn.Linear(in_features, resqu_features, bias=resqu_bias)
      nn.init.xavier_uniform_(self.resqu_linear.weight, 0.2)

    self.norm1 = nn.BatchNorm1d(relu_features) if norm else nn.Identity()
    self.norm2 = nn.BatchNorm1d(resqu_features) if norm else nn.Identity()
    self.bound = BatchBound() if bound_resqu else nn.Identity()

  def forward(self, x):
    if self.resqu_linear:
      z = torch.pow(self.resqu_linear(x), 2)
      x2 = torch.pow(x, 2)
      v = F.linear(x2, torch.pow(self.resqu_linear.weight, 2), torch.pow(self.resqu_linear.bias, 2))
      s = z - v
      oc = torch.cat((self.norm1(self.relu_linear(x)), self.bound(self.norm2(s))), dim=1)
      return F.relu(oc)
    else:
      return F.relu(self.relu_linear(x))




class SplitReSqUConv(nn.Module):
  def __init__(self, in_channels, relu_channels, resqu_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, resqu_bias=True, norm=True, bound_resqu=True):
    super(SplitReSqUConv, self).__init__()
    self.in_channels, self.relu_channels, self.resque_channels = in_channels, relu_channels, resqu_channels
    self.convr = None
    if resqu_channels != 0:
      self.convr = nn.Conv2d(self.in_channels, self.resque_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=resqu_bias)
      nn.init.xavier_uniform_(self.convr.weight, 0.2)

    self.conv = nn.Conv2d(self.in_channels, self.relu_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    
    self.norm1 = nn.BatchNorm2d(relu_channels) if norm else nn.Identity()
    self.norm2 = nn.BatchNorm2d(resqu_channels) if norm else nn.Identity()
    self.bound = BatchBound() if bound_resqu else nn.Identity()

  def forward(self, x):
    if self.convr:
      z = torch.pow(self.convr(x), 2)
      x2 = torch.pow(x, 2)
      v = F.conv2d(x2, torch.pow(self.convr.weight, 2), torch.pow(self.convr.bias, 2), self.convr.stride, padding=self.convr.padding, dilation=self.convr.dilation, groups=self.convr.groups)
      s = z - v
      # oc = torch.cat((self.norm(self.conv(x)), self.norm(s)), dim=1)
      oc = torch.cat((self.norm1(self.conv(x)), self.bound(self.norm2(s))), dim=1)
      o = F.relu(oc)
      return o
    else:
      return F.relu(self.conv(x))
    
  def resqu_wb_norm(self):
    return torch.linalg.vector_norm(self.convr.weight), torch.linalg.vector_norm(self.convr.bias)



# normalizes values into a range of -1 and 1, with optional polarization away from 0
class BatchBound(nn.Module):

  def __init__(self, gamma=0.9, polarization=0):
    super(BatchBound, self).__init__()
    # self.max = None
    self.polarization = polarization

  def forward(self, x):

    tx = torch.pow(x, 1 / (2 * self.polarization + 1))
    maxi = torch.amax(torch.abs(tx), dim=0)
    # print(maxi.shape)
    return tx / maxi

class ChannelBound(nn.Module):
  def __init__(self, polarization=0):
    super(ChannelBound, self).__init__()
    self.polarization = polarization

  def forward(self, x):
    tx = torch.pow(x, 1 / (2 * self.polarization + 1))
    maxi = torch.amax(torch.abs(tx), dim=(-1, -2), keepdim=True)
    # maxi = maxi.view(*maxi.shape, 1, 1).expand_as(x)
    return tx / maxi

  
