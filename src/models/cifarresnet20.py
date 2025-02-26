import torch
import torch.nn as nn
import torch.nn.functional as F
from ..nn_modules.activations import ReSqU
from ..nn_modules.layers import MultiActivationDense
from ..nn_modules.layers import MultiActivationConv
from ..nn_modules.layers import VarianceSplitConv
from ..nn_modules.trainingmodel import TrainingModel

class TestNet(TrainingModel):
  def __init__(self):
    super().__init__()
    self.conv11 = VarianceSplitConv(3, [(nn.ReLU(), 20), (ReSqU(), 3)], 3, padding=1)
    self.conv12 = VarianceSplitConv(24, [(nn.ReLU(), 20), (ReSqU(), 3)], 3, padding=1)
    self.conv13 = nn.Conv2d(24, 24, 3, padding=1)

    self.conv21 = VarianceSplitConv(24, [(nn.ReLU(), 20), (ReSqU(), 3)], 3, padding=1)
    self.conv22 = nn.Conv2d(24, 24, 3, padding=1)
    
    self.conv31 = VarianceSplitConv(24, [(nn.ReLU(), 20), (ReSqU(), 3)], 3, padding=1)
    self.conv32 = nn.Conv2d(24, 24, 3, padding=1)

    self.conv41 = VarianceSplitConv(24, [(nn.ReLU(), 20), (ReSqU(), 3)], 3, padding=1)
    self.conv42 = nn.Conv2d(24, 24, 3, padding=1)

    self.conv51 = VarianceSplitConv(24, [(nn.ReLU(), 28), (ReSqU(), 3)], 17)
    self.conv52 = VarianceSplitConv(32, [(nn.ReLU(), 28), (ReSqU(), 3)], 3, padding=1)
    self.conv53 = nn.Conv2d(24, 32, 17)

    self.conv61 = VarianceSplitConv(32, [(nn.ReLU(), 28), (ReSqU(), 3)], 3, padding=1)
    self.conv62 = nn.Conv2d(32, 32, 3, padding=1)

    # self.conv71 = MultiActivationConv(32, [(nn.ReLU(), 28), (ReSqU(), 4)], 3)
    # self.conv72 = MultiActivationConv(32, [(nn.ReLU(), 28), (ReSqU(), 4)], 3)
    # self.conv73 = MultiActivationConv(32, [(nn.ReLU(), 28), (ReSqU(), 4)], 3)

    self.pool = nn.AvgPool2d(8,4)
    self.fc1a = MultiActivationDense(288, [(nn.ReLU(), 40), (ReSqU(), 10)])
    self.fco = nn.Linear(50, 10)
    

  def forward(self, x):
    # print('input', torch.max(x))
    x1 = self.conv11(x)
    # print('max', torch.max(x1), 'max param', torch.max(self.conv11.conv.weight))
    x = self.conv12(x1)
    # print('max', torch.max(x), 'max param', torch.max(self.conv12.conv.weight))
    x = F.relu(torch.add(self.conv13(x), x1))
    # print(torch.isnan(x).any())
    # print('max', torch.max(x))
    x1 = x    
    x = self.conv21(x)
    x = F.relu(self.conv22(x) + x1)
    # print(torch.isnan(x).any())
    # print('max', torch.max(x))

    x1 = x    
    x = self.conv31(x)
    x = F.relu(self.conv32(x) + x1)
    
    x1 = x    
    x = self.conv41(x)
    x = F.relu(self.conv42(x) + x1)
    
    x1 = x
    x = self.conv51(x1)
    x = F.relu(self.conv52(x) + self.conv53(x1))

    x1 = x
    x = self.conv61(x1)
    x = F.relu(self.conv62(x) + x1)

    x = self.pool(x)

    x = torch.flatten(x, 1) # flatten all dimensions except batch
    x = self.fc1a(x)
    x = self.fco(x)
    return F.log_softmax(x, dim=1)