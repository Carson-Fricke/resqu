import torch
import torch.nn as nn
import torch.nn.functional as F

class ReSqU(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return 0.5 * torch.pow(F.relu(x), 2)
  
class VarA(nn.Module):
  def __init__(self):
    super().__init__()
  
  def forward(self, x):
    return torch.sum(torch.pow(F.relu(x), 2))

# Polynomial Adaptive Rectified Unit
class PAReU(nn.Module):
  size: int
  def __init__(self, ip_size):
    super().__init__()
    self.size = ip_size
    print(ip_size)
    exponents = torch.zeros(self.size)
    self.params = nn.Parameter(exponents)
    torch.nn.init.uniform_(self.params, -2, 2)

  def forward(self, x):
    return torch.pow(F.relu(x), torch.add(F.tanh(self.params), 1))


class SharedActivation(nn.Module):
  def __init__(self, activations):
    super().__init__()
    self.activations = activations
    # self.weights = torch.zeros((1, len(self.activations), 1))
    self.weights = torch.Tensor([[[1 / (2*n+1)] for n in range(len(activations))]])
    self.params = nn.Parameter(self.weights)

  def forward(self, x):
    in_shape = x.shape
    al = [activation(x).flatten() for activation in self.activations]
    a = torch.stack(al)
    s2 = torch.pow(self.params, 2)
    o = F.conv1d(a, s2 / torch.sum(s2))
    return o.reshape(in_shape)  