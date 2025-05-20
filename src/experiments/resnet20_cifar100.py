import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.optim as optim
from src.models.resnet import resnet20
from src.nn_modules.trainingmodel import FixedCrossEntropy
import os

def run_experiment(seed=777, n=5):
  torch.manual_seed(seed)
  torch.set_printoptions(precision=6, sci_mode=False)
  
  stats = ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
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

  batch_size = 100

  # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  trainset50k = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
  testset10k = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_train)
  fullset = torch.utils.data.ConcatDataset([trainset50k, testset10k])
  trainset, testset = torch.utils.data.random_split(fullset, [55000, 5000])

  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

  clipping_strategy = lambda params, epochs: torch.nn.utils.clip_grad_norm_(params, 3) #if epochs < 50 else torch.nn.utils.clip_grad_value_(params, 0.3)

  for run in range(n):
    for rate in [0, 3/64, 3/32, 3/16, 1/2]:
      os.makedirs(f'resnet20-CIFAR100/rate{rate:.3f}', exist_ok=True)
      net = resnet20(resqu_rate=rate, num_classes=100).to('cuda')
      criterion = FixedCrossEntropy(reduction='batchmean')
      optimizer = optim.SGD(net.parameters(), momentum=0.9, weight_decay=0.000015)
      torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.1, epochs=200, steps_per_epoch=550, pct_start=0.1, div_factor=25, final_div_factor=300, three_phase=True)
      print('Parameter Count: ', sum(p.numel() for p in net.parameters()))
      net.fit(trainloader, testloader, optimizer, criterion, epochs=200, grad_clipper=clipping_strategy, save_options=(None, f'./resnet20-CIFAR100/rate{rate:.3f}/resnet20-CIFAR100-rr-{rate:.3f}-run-{run}-'))
