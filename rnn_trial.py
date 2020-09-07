
"""
Created on Fri Jul 24 22:28:41 2020

@author: Sivaraman Sivaraj
"""


#import
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#setting the device
device = torch.device('cpu')

#hyper parameters
input_size = 28
num_classess = 10
num_layers = 2
hidden_size = 256
sequence_length = 28
learning_rate = 0.001
batch_size = 64
num_epoch = 1

#creating the RNN


class RNN(nn.Module):
    def __init__(self, input_size,hidden_size,num_layers,num_classess):
        super(RNN,self).__init__()
        self.hidden_size = hidden_size
        self.num_classess = num_classess
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size,num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classess)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        #forward propagation
        out, _  = self.rnn(x,h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        
        return out
    
#download the data

train_dataset = datasets.MNIST(root = 'dataset/', train = True, 
                               transform = transforms.ToTensor(), download = True)
test_dataset = datasets.MNIST(root = 'dataset/', train = False, 
                               transform = transforms.ToTensor(), download = True)

train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

#initialize network

model = RNN(input_size, hidden_size, num_layers, num_classess).to(device)

#loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

       
for epoch in range(num_epoch):
    for batch_idx, (data, targets) in enumerate(train_loader):
        #Get the data to cuda if possible
        data = data.to(device = device).squeeze(1)
        # print(data.shape)
        targets = targets.to(device = device)
       
        # #get the correct shape
        # data = data.reshape(data.shape[0], -1)
        
        #forward
        scores = model(data)
        # print(scores.shape)
        loss = criterion(scores, targets)
        
        #back propagation
        optimizer.zero_grad()
        loss.backward()
        
        #gradient descent or optimizer
        optimizer.step()
        
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Cheacking accuracy on training data")
    else:
        print("Checking accuracy on test data")
        
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device = device).squeeze(1)
            y = y.to(device = device)
            
            scores = model(x)
            _, predictions = scores.max(1)
            # print(predictions)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            
        print(f'Got {num_correct}/ {num_samples} with accuracy \
              {float(num_correct)/float(num_samples)*100:.2f}')
    model.train()
    
    
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)


            
        
      
        
        
  























