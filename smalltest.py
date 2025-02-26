import torch
import torchvision
import torchvision.transforms.v2 as transforms
from src.nn_modules.layers import *
from src.nn_modules.activations import *
import torch.optim as optim
import torch.nn as nn
from torch import autograd

if __name__ == '__main__':

  stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
  transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToImage(), 
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(*stats)
  ])

  transform_train = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomRotation(15),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.ToImage(), 
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(*stats, inplace=True)
  ])

  batch_size = 100

  trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform_train)

  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

  testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                           shuffle=False, num_workers=2)
  
  i, (x, labels) = list(enumerate(trainloader, 0))[0]

  torch.manual_seed(1)
  l = VarianceSplitConv(3, [(nn.ReLU(), 10), (ReSqU(), 5)], 3)

  print(l(x))