import torch
import torchvision
import torchvision.transforms.v2 as transforms
from src.models.cifardensenet65k import *
import torch.optim as optim
import torch.nn as nn
from torch import autograd
from src.models.resnet20 import *
from src.nn_modules.trainingmodel import FixedCrossEntropy

torch.manual_seed(1)
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
    # transforms.RandomRotation(15),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToImage(), 
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(*stats, inplace=True)
  ])

  batch_size = 64

  trainset50k = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform_train)
  
  testset10k = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform_train)

  fullset = torch.utils.data.ConcatDataset([trainset50k, testset10k])

  trainset, testset = torch.utils.data.random_split(fullset, [55000, 5000])

  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=4, pin_memory=True)

  
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                           shuffle=False, num_workers=4)

  classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



  net = TestNet().to('cuda')
  # criterion = nn.MSELoss()
  criterion = FixedCrossEntropy(reduction='batchmean')
  optimizer = optim.SGD(net.parameters(), lr=0.001, weight_decay=0.000015)
  # optimizer = optim.AdamW(net.parameters())
  
  # torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.5, epochs=30, steps_per_epoch=500, div_factor=500, final_div_factor=100, three_phase=True)
  # SGD scheduler
  torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.1, epochs=500, steps_per_epoch=500, pct_start=0.05, div_factor=25, final_div_factor=1000, three_phase=True)
  print('Parameter Count: ', sum(p.numel() for p in net.parameters()))
  # torch.optim.lr_scheduler.MultiStepLR()

  torch.set_printoptions(precision=4, sci_mode=False)
  
  net.fit(trainloader, testloader, optimizer, criterion, epochs=500, save_options=(None, 'model'))
