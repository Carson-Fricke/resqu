import os
import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.optim as optim
# import torch_optimizer as optim
import torch.nn as nn
from src.models.single_var_regresser import SingleVarRegresser
from src.models.resnet import *  
from src.synthetic_datasets.test_datasets import SineWave


def run_experiment(seed=777, n=3):
 

  batch_size = 100

  root = SineWave(input_range=(-8,8), noise=0.5)

  trainset, testset = torch.utils.data.random_split(root, (1200, 400))

  # fullset = torch.utils.data.ConcatDataset([trainset60k, testset10k])

  # trainset, testset = torch.utils.data.random_split(fullset, [65000, 5000])

  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
  
  clipping_strategy = lambda params, epochs: torch.nn.utils.clip_grad_norm_(params, 4) #if epochs < 30 else torch.nn.utils.clip_grad_value_(params, 0.5)
  
  for run in range(n):
    for rate in [ 0, 1/8, 1/4, 1/2 ]:
      os.makedirs(f'single-var-sine/rate{rate:.3f}', exist_ok=True)
      net = SingleVarRegresser(8,2,resqu_rate=rate).to('cuda')
      # criterion = nn.MSELoss()
      criterion = nn.MSELoss(reduction='sum')
      # optimizer = optim.SGD(net.parameters(), momentum=0.9, weight_decay=0.000015)
      optimizer = optim.SGD(net.parameters())
      # torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 1365, 2, 0.0001)
      torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.4, epochs=20, steps_per_epoch=12, pct_start=0.1, div_factor=500, final_div_factor=300, three_phase=True)
      # torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 1365, 2, 0.0001)
      # torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.1, epochs=100, steps_per_epoch=430, pct_start=0.05, div_factor=25, final_div_factor=300, three_phase=True)
      print('Parameter Count: ', sum(p.numel() for p in net.parameters()))
      
      torch.set_printoptions(precision=6, sci_mode=False)
      # loader_transform=lambda x: (x[0], x[1].unsqueeze(-1)),
      net.fit(trainloader, testloader, optimizer, criterion, epochs=20, loader_transform=lambda x: (x[0].unsqueeze(-1), x[1].unsqueeze(-1)), grad_clipper=clipping_strategy, save_options=(5, f'./single-var-sine/rate{rate:.3f}/single-var-sine-rr-{rate:.3f}-run-{run}-'))