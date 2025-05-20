import os
import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.optim as optim
# import torch_optimizer as optim
import torch.nn as nn
from src.models.linear_autoencoder import LinearAutoEncoder
from src.models.resnet import *  


def run_experiment(seed=777, n=3):
  stats = ((0.1307,), (0.3081,))
  transform = transforms.Compose([
    torchvision.transforms.ToTensor(),
    transforms.Resize((28,28)),
    transforms.ToImage(), 
    transforms.ToDtype(torch.float32, scale=True),
    # transforms.Normalize(*stats)
  ])

  transform_train = transforms.Compose([
    torchvision.transforms.ToTensor(),
    transforms.Resize((28,28)),
    transforms.RandomRotation(20),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    # transforms.RandomCrop(28, padding=4),
    transforms.ToImage(), 
    transforms.ToDtype(torch.float32, scale=True),
    # transforms.Normalize(*stats, inplace=True)
  ])

  batch_size = 100

  trainset60k = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
  
  testset10k = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_train)

  # fullset = torch.utils.data.ConcatDataset([trainset60k, testset10k])

  # trainset, testset = torch.utils.data.random_split(fullset, [65000, 5000])

  trainloader = torch.utils.data.DataLoader(trainset60k, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
  testloader = torch.utils.data.DataLoader(testset10k, batch_size=batch_size, shuffle=False, num_workers=4)
  
  clipping_strategy = lambda params, epochs: torch.nn.utils.clip_grad_norm_(params, 4) #if epochs < 30 else torch.nn.utils.clip_grad_value_(params, 0.5)
  
  for run in range(n):
    for rate in [0, 1/16, 1/8, 1/4, 1/2]:
      os.makedirs(f'linear-autoencoder-mnist/rate{rate:.3f}', exist_ok=True)
      net = LinearAutoEncoder((28,28), 1, resqu_rate=rate).to('cuda')
      # criterion = nn.MSELoss()
      criterion = nn.MSELoss(reduction='sum')
      # optimizer = optim.SGD(net.parameters(), momentum=0.9, weight_decay=0.000015)
      optimizer = optim.SGD(net.parameters())
      # torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 1365, 2, 0.0001)
      torch.optim.lr_scheduler.OneCycleLR(optimizer, 2, epochs=100, steps_per_epoch=600, pct_start=0.1, div_factor=500, final_div_factor=300, three_phase=True)
      # torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 1365, 2, 0.0001)
      # torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.1, epochs=100, steps_per_epoch=430, pct_start=0.05, div_factor=25, final_div_factor=300, three_phase=True)
      print('Parameter Count: ', sum(p.numel() for p in net.parameters()))
      
      torch.set_printoptions(precision=6, sci_mode=False)

      net.fit(trainloader, testloader, optimizer, criterion, epochs=100, grad_clipper=clipping_strategy, loader_transform=lambda x: (x[0].squeeze(), x[0].squeeze()), save_options=(5, f'./linear-autoencoder-mnist/rate{rate:.3f}/linear-autoencoder-mnist-rr-{rate:.3f}-run-{run}-'))