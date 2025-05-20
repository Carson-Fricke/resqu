import torch
import torchvision
import torchvision.transforms.v2 as transforms
# import torch_optimizer as optim
import torch.optim as optim
from src.models.resnet import resnet20
from src.nn_modules.trainingmodel import FixedCrossEntropy
import os

def run_experiment(seed=777, n=3):
  torch.manual_seed(seed)
  torch.set_printoptions(precision=6, sci_mode=False)
  
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

  batch_size = 128

  # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  trainset50k = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
  testset10k = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_train)
  fullset = torch.utils.data.ConcatDataset([trainset50k, testset10k])
  trainset, testset = torch.utils.data.random_split(fullset, [55000, 5000])

  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

  clipping_strategy = lambda params, epochs: torch.nn.utils.clip_grad_norm_(params, 4) #else torch.nn.utils.clip_grad_value_(params, 0.5)

  for run in range(n):
    for rate in [0, 3/64, 3/32, 3/16, 1/2]:
      os.makedirs(f'resnet20-CIFAR10/rate{rate:.3f}', exist_ok=True)
      torch.manual_seed(run)
      net = resnet20(resqu_rate=rate).to('cuda')
      criterion = FixedCrossEntropy(reduction='sum')
      optimizer = optim.SGD(net.parameters(), lr=0.12, momentum=0.9, weight_decay=0.00001)
      # torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.12, epochs=200, steps_per_epoch=430, pct_start=0.001, div_factor=25, final_div_factor=200, three_phase=True)
      # torch.optim.lr_scheduler.StepLR(optimizer, step_size=25800)
      torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200*430, 0.001)
      # torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 1365, 2, 0.0001)
      print('Parameter Count: ', sum(p.numel() for p in net.parameters()))
      net.fit(trainloader, testloader, optimizer, criterion, epochs=200, grad_clipper=clipping_strategy, save_options=(None, f'./resnet20-CIFAR10/rate{rate:.3f}/resnet20-CIFAR10-rr-{rate:.3f}-run-{run}-'))
