import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.optim as optim
# import torch_optimizer as optim
import torch.nn as nn
from src.models.mnist_autoencoder import MnistAutoEncoder
from src.models.resnet20 import *



torch.manual_seed(777)
if __name__ == '__main__':

  

  stats = ((0.1307,), (0.3081,))
  transform = transforms.Compose([
    torchvision.transforms.ToTensor(),
    transforms.Resize((28,28)),
    transforms.ToImage(), 
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(*stats)
  ])

  transform_train = transforms.Compose([
    torchvision.transforms.ToTensor(),
    transforms.Resize((28,28)),
    transforms.RandomRotation(20),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomCrop(28, padding=4),
    transforms.ToImage(), 
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(*stats, inplace=True)
  ])

  batch_size = 100

  trainset50k = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
  
  testset10k = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_train)

  fullset = torch.utils.data.ConcatDataset([trainset50k, testset10k])

  trainset, testset = torch.utils.data.random_split(fullset, [65000, 5000])

  

  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)

  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)


  net = MnistAutoEncoder(resqu_rate=3/32).to('cuda')
  # criterion = nn.MSELoss()
  criterion = nn.MSELoss(reduction='mean')
  optimizer = optim.SGD(net.parameters(), momentum=0.9, weight_decay=0.000015)
  # optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.000015)
  
  # AdamW scheduler
  # torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.5, epochs=30, steps_per_epoch=550, pct_start=0.6, div_factor=5000, final_div_factor=500, three_phase=True)
  # SGD scheduler
  torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.1, epochs=500, steps_per_epoch=550, pct_start=0.05, div_factor=25, final_div_factor=300, three_phase=True)
  print('Parameter Count: ', sum(p.numel() for p in net.parameters()))
  # torch.optim.lr_scheduler.MultiStepLR()

  torch.set_printoptions(precision=6, sci_mode=False)
  
  net.fit(trainloader, testloader, optimizer, criterion, epochs=500, loader_transform=lambda x: (x[0].squeeze(), x[0].squeeze()), save_options=(5, 'MNIST_autoencoder'))
