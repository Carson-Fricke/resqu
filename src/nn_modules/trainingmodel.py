import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from datetime import datetime, timezone
from torch.utils.data import Dataset
import pandas as pd


class FixedCrossEntropy(nn.Module):
  def __init__(self, reduction='batchmean'):
    super(FixedCrossEntropy, self).__init__()
    self.losser = nn.KLDivLoss(reduction=reduction)
  
  def forward(self, x, labels):
    num_classes = x.size(-1)
    one_hot = (0.9 * F.one_hot(labels, num_classes=num_classes) + 0.05).to(x.device, dtype=torch.float32)
    return self.losser(x, one_hot)


class TrainingModel(nn.Module):

  def __init__(self):
    super(TrainingModel, self).__init__()

  def forward(self, x):
    raise NotImplementedError()
  
  def fit(self, training_loader, test_loader, optimizer, criterion, epochs=150, loader_transform=lambda x: x, print_options=(True, 1, 1), save_options=(None, 'save')):

        
    np = print_options[1]
    pe = len(training_loader) / np

    best_correct = 0

    start_time = int(round(datetime.now(timezone.utc).timestamp()*1000))

    tsc = ['epoch', 'batch', 'time_stamp', 'train_loss', 'val_loss', 'val_accuracy']
    training_stats = pd.DataFrame(columns=tsc)
    
    try:
      for epoch in range(epochs):  # loop over the dataset multiple times
      
        if (save_options[0] and epoch % save_options[0] == 0):
          torch.save(self.state_dict(), f'{save_options[1]}{start_time}.th')

        #  train the model
        running_loss = 0.0
        for i, data in enumerate(training_loader, 0):
          # get the inputs; data is a list of [inputs, labels]
          x, y = loader_transform(data)
          if torch.isnan(x).any():
            print('Input contains nan', x)
          # zero the parameter gradients
          optimizer.zero_grad()

          # with torch.autograd.set_detect_anomaly(True):
          # forward + backward + optimize
          # with autograd.detect_anomaly():
          outputs = self.forward(x.to('cuda'))
          # loss = criterion(outputs, (0.9 * F.one_hot(labels, num_classes=num_classes) + 0.05).to('cuda', dtype=torch.float32))
          loss = criterion(outputs, y.to('cuda'))

          loss.backward()
          nn.utils.clip_grad_norm_(self.parameters(), 3)
          # nn.utils.clip_grad_value_(self.parameters(), 0.5)
          optimizer.step()
          # print statistics
          running_loss += loss.item()
          if i % pe == pe - 1:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / pe:.5f}')

        # evaluate with test data
        if (print_options[0] and epoch % print_options[2] == 0):
          test_loss = 0.0
          correct = 0
          total = 0
          with torch.no_grad():
            for data in test_loader:
              x, y = loader_transform(data)
              # calculate outputs by running images through the network
              outputs = self.forward(x.to('cuda'))
              
              test_loss += criterion(outputs, y.to('cuda')).item()
              # the class with the highest energy is what we choose as prediction
              if x.shape != y.shape:
                _, predicted = torch.max(outputs, 1)
                correct += (predicted.cpu() == y).sum().item()
              total += y.size(0)

            time_now = int(round(datetime.now(timezone.utc).timestamp()*1000)) - start_time
            measurement = pd.DataFrame([[epoch, 0, time_now, running_loss / pe, y.size(0) * test_loss / total, correct / total]], columns=tsc)
            training_stats = pd.concat([training_stats, measurement], ignore_index=True)

            if correct > best_correct:
              best_correct = correct
              torch.save(self.state_dict(), f'{save_options[1]}{start_time}.th')

            print(f'Accuracy of the network on the {total} validation images: {100 * correct // total} %')

        
    finally:
      training_stats.to_csv(f'{save_options[1]}{str(start_time)[:10]}_training_stats.csv')
              
    print('Finished Training')

  


