import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nn_modules.layers import BatchBound, SplitReSqUConv

from ..nn_modules.trainingmodel import TrainingModel

from math import floor, ceil


def _weights_init(m):
  """
      Initialization of CNN weights
  """
  classname = m.__class__.__name__
  if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
    nn.init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    """
      Identity mapping between ResNet blocks with diffrenet size feature map
    """
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, in_planes, planes, stride=1, option='A', force_bound=False):
    use_bb = False# (not planes[1] == 0) or force_bound

    super(BasicBlock, self).__init__()
    self.conv1 = SplitReSqUConv(in_planes, *planes, kernel_size=3, stride=stride, padding=1)
    self.bn1 = nn.BatchNorm2d(sum(planes))
    self.bb1 = BatchBound() if use_bb else nn.Identity()
    self.conv2 = SplitReSqUConv(sum(planes), *planes, kernel_size=3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(sum(planes))
    self.bb2 = BatchBound() if use_bb else nn.Identity()
    self.shortcut = nn.Identity()
    if stride != 1 or in_planes != sum(planes):
      self.shortcut = LambdaLayer(
        lambda x:
          F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, sum(planes)//4, sum(planes)//4), "constant", 0)
      )

  def forward(self, x):
    out = F.relu(self.bb1(self.bn1(self.conv1(x))))
    out = self.bb2(self.bn2(self.conv2(out)))
    s = self.shortcut(x)
    out += s
    out = F.relu(out)
    return out
    
class ResNet(TrainingModel):
  
  def __init__(self, block, num_blocks, resqu_rate=0, in_planes=3, num_classes=10, force_bound=False):
    super(ResNet, self).__init__()
    self.intermediate_planes = 16
    self.conv1 = nn.Conv2d(in_planes, 16, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(16)
    relu_rate = 1 - resqu_rate

    self.num_classes = num_classes

    self.layer1 = self._make_layer(block, (floor(16 * relu_rate), ceil(16 * resqu_rate)), num_blocks[0], stride=1, force_bound=force_bound)
    self.layer2 = self._make_layer(block, (floor(32 * relu_rate), ceil(32 * resqu_rate)), num_blocks[1], stride=2, force_bound=force_bound)
    self.layer3 = self._make_layer(block, (floor(64 * relu_rate), ceil(64 * resqu_rate)), num_blocks[2], stride=2, force_bound=force_bound)
    self.linear = None
    self.apply(_weights_init)

  def _make_layer(self, block, planes, num_blocks, stride, force_bound=False):
    strides = [stride] + [1]*(num_blocks-1)
    layers = []
    for stride in strides:
      layers.append(block(self.intermediate_planes, planes, stride, force_bound=force_bound))
      self.intermediate_planes = sum(planes) * block.expansion
    return nn.Sequential(*layers)

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = F.avg_pool2d(out, out.size()[3])
    out = out.view(out.size(0), -1)
    if not self.linear:
      self.linear = nn.Linear(out.size(-1), self.num_classes).to(out.device)
    out = self.linear(out)
    return F.log_softmax(out, dim=1)

def resnet17(resqu_rate=0, in_planes=3, num_classes=10, force_bound=False):
    return ResNet(BasicBlock, [2, 2, 2], resqu_rate=resqu_rate, in_planes=in_planes, num_classes=num_classes, force_bound=force_bound)

def resnet20(resqu_rate=0, in_planes=3, num_classes=10, force_bound=False):
    return ResNet(BasicBlock, [3, 3, 3], resqu_rate=resqu_rate, in_planes=in_planes, num_classes=num_classes, force_bound=force_bound)

def resnet44(resqu_rate=0, in_planes=3, num_classes=10, force_bound=False):
    return ResNet(BasicBlock, [7, 7, 7], resqu_rate=resqu_rate, in_planes=in_planes, num_classes=num_classes, force_bound=force_bound)