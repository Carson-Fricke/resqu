import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd


class FixedCrossEntropy(nn.Module):
  def __init__(self):
    super(FixedCrossEntropy, self).__init__()



class TrainingModel(nn.Module):

  def __init__(self):
    super(TrainingModel, self).__init__()

  def forward(self, x):
    raise NotImplementedError()
  
  def fit(self, training_loader, test_loader, optimizer, criterion, epochs=150,  print_options=(True, 1, 3), save_options=(5, 'save.th')):

    np = print_options[1]
    pe = len(training_loader) / np

    for epoch in range(epochs):  # loop over the dataset multiple times
    
      if (epoch % save_options[0] == 0):
        torch.save(self.state_dict(), save_options[1])

      if (print_options[0] and epoch % print_options[2] == 0):
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
          for data in test_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = self.forward(images.to('cuda'))
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum().item()
          print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
      #   state_dict = torch.load('...')
      #   net.load_state_dict(state_dict['model'])
      #   optimizer = post_optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, nesterov=True, weight_decay=0.0002)
      running_loss = 0.0
      for i, data in enumerate(training_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        if torch.isnan(inputs).any():
          print('Input contains nan', inputs)
        # zero the parameter gradients
        optimizer.zero_grad()

        with torch.autograd.set_detect_anomaly(True):
        # forward + backward + optimize
        # with autograd.detect_anomaly():
          outputs = self.forward(inputs.to('cuda'))
          loss = criterion(outputs, (0.8 * F.one_hot(labels, num_classes=10) + 0.1).to('cuda', dtype=torch.float32))
        # loss = criterion(outputs.to('cuda'), labels.to('cuda'))
        
        # print(loss)
          loss.backward()
          nn.utils.clip_grad_value_(self.parameters(), 0.05)
          optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % pe == pe - 1:    # print every 2000 mini-batches
          print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / pe:.3f}')
          
              
    print('Finished Training')

  


