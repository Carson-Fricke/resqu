import torch
import torch.nn as nn
import torch.nn.functional as F
from ..nn_modules.activations import ReSqU
from ..nn_modules.layers import MultiActivationDense
from ..nn_modules.layers import InstanceUniform
from ..nn_modules.layers import SplitReSqUConv
from ..nn_modules.trainingmodel import TrainingModel


class DenseBlock(nn.Module):
  def __init__(self, in_channels=16, relu_channels=14, resqu_channels=2, n=3):
    super(ResBlock, self).__init__()    
    self.layers = [SplitReSqUConv(in_channels, relu_channels=relu_channels, resqu_channels=resqu_channels, kernel_size=3, padding=1,).to('cuda') for _ in range(n)]
  
  def forward(self, x):
    lp = [x] + [None for l in self.layers]
    a = x
    for i, l in enumerate(self.layers):
      a = F.leaky_relu(a + l(a))
      # lp[i+1] = l(lp[i])
    # o = x + lp[-1]
    return a

class ResBlock(nn.Module):
  
  def __init__(self, in_channels=16, relu_channels=14, resqu_channels=2, n=1):
    super(ResBlock, self).__init__()    
    self.layers = [SplitReSqUConv(in_channels, relu_channels=relu_channels, resqu_channels=resqu_channels, kernel_size=3, padding=1,).to('cuda') for _ in range(n)]
  
  def forward(self, x):
    lp = [x] + [None for l in self.layers]
    
    for i, l in enumerate(self.layers):
      lp[i+1] = l(lp[i])
    o = x + lp[-1]
    return F.leaky_relu(o)
    
class SquishBlock(nn.Module):
  def __init__(self, in_channels=16, relu_channels=30, resqu_channels=2, kernel=5, padding=0, ):
    super(SquishBlock, self).__init__()
    out_channels = relu_channels + resqu_channels
    self.conv1 = SplitReSqUConv(in_channels, relu_channels, resqu_channels, kernel, padding=padding).to('cuda')
    self.conv2 = nn.Conv2d(in_channels, out_channels, kernel, padding=padding, bias=False).to('cuda')
    nn.init.xavier_uniform_(self.conv2.weight)
  def forward(self, x):
    xo = self.conv1(x) + self.conv2(x)
    return F.leaky_relu(xo)



class TestNet(TrainingModel):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
    nn.init.xavier_uniform_(self.conv1.weight, 0.5)
    self.norm1 = nn.Identity() #InstanceUniform()

    self.res1 = ResBlock(n=6)
    # self.res2 = ResBlock()
    # self.res3 = ResBlock()
    # self.res4 = ResBlock()

    self.squish1 = SquishBlock(kernel=7)

    # self.res21 = ResBlock(in_channels=32, activations=[(nn.ReLU(), 29), (ReSqU(), 2)])
    # self.res22 = ResBlock(in_channels=32, activations=[(nn.ReLU(), 29), (ReSqU(), 2)])
    # self.res23 = ResBlock(in_channels=32, activations=[(nn.ReLU(), 29), (ReSqU(), 2)])
    # self.res24 = ResBlock(in_channels=32, activations=[(nn.ReLU(), 29), (ReSqU(), 2)])

    # self.squish2 = SquishBlock(in_channels=32, kernel=7, activations=[(nn.ReLU(), 124), (ReSqU(), 3)])

    # self.res31 = ResBlock(in_channels=128, activations=[(nn.ReLU(), 124), (ReSqU(), 3)])
    # self.res32 = ResBlock(in_channels=128, activations=[(nn.ReLU(), 124), (ReSqU(), 3)])
    # self.res33 = ResBlock(in_channels=128, activations=[(nn.ReLU(), 124), (ReSqU(), 3)])
    # self.res34 = ResBlock(in_channels=128, activations=[(nn.ReLU(), 124), (ReSqU(), 3)])

    # self.squish3 = SquishBlock(in_channels=128, kernel=9, activations=[(nn.ReLU(), 251), (ReSqU(), 4)])

    self.pool = nn.AvgPool2d(8,8)
    self.fc1a = MultiActivationDense(288, [(nn.ReLU(), 48), (ReSqU(), 2)])
    self.fco = nn.Linear(50, 10)
    

  def forward(self, x):
    
    # print('input', torch.max(x))
    x = self.conv1(x)
    # print('c1, max', torch.max(x), 'mean, stdev', torch.std_mean(x),'max params', torch.max(self.conv1.weight))
    x = self.norm1(x)
    # print('n1, max', torch.max(x), 'mean, stdev', torch.std_mean(x),)
    x = self.res1(x)
    # print('r1, max', torch.max(x), torch.argmax(x))
    # x = self.res2(x)
    # print('r2, max', torch.max(x), torch.argmax(x))
    # x = self.res3(x)
    # # print('r3, max', torch.max(x), torch.argmax(x))
    # x = self.res4(x)
    # print('r4, max', torch.max(x), torch.argmax(x))
        
    x = self.squish1(x)
    # print('squish: max', torch.max(x), torch.argmax(x))

    # x = self.res21(x)
    # x = self.res22(x)
    # x = self.res23(x)
    # x = self.res24(x)

    # x = self.squish2(x)

    # x = self.res31(x)
    # x = self.res32(x)
    # x = self.res33(x)
    # x = self.res34(x)

    # x = self.squish3(x)

    x = self.pool(x)
    # print('poolo: max', torch.max(x), torch.argmax(x))
    x = torch.flatten(x, 1) # flatten all dimensions except batch
    x = self.fc1a(x)
    x = self.fco(x)
    # print('output: max', torch.max(x), torch.argmax(x))
    return F.log_softmax(x, dim=1)