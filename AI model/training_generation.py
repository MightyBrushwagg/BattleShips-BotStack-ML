import numpy as np
import math
from boardGeneration import Board
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F


# code flow

# generate lots of boards, store them appropriately
# input: hit boards. Correct: ship placement with negative shots

# define model (both pytorch and tensorflor)

# train model

class Data_Set(Dataset):
    def __init__(self):
        # super().__init__()
        self.n = np.random.uniform(0,40)

    def __len__(self):
        # this is total training number
        return 10000
    
    def __getitem__(self, index):
        # generate random board with random n 
        
        battleship = Board(n=self.n)
        hits = torch.tensor(battleship.getHits()).unsqueeze(0).float()
        ships = torch.tensor(battleship.getShipPlacement()).view(-1).float()
        
        return hits, ships
    

# --- Model ---
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=4, padding=1) # output is shape: [8, 9, 9]
        self.conv2 = nn.Conv2d(8, 16, kernel_size=4, padding=1) # output is shape: [16, 8, 8]
        self.conv3 = nn.Conv2d(16, 32, kernel_size=4, padding=1) # output is shape: [32, 7, 7]
        self.pool = nn.MaxPool2d(2, 2) # output shape has 256 neurons (print(x.shape) after self.pool to find shape)

        self.fc1 = nn.Linear(256, 150)
        self.fc2 = nn.Linear(150, 84)
        self.fc3 = nn.Linear(84, 100)  # 100 = 10x10 ship placement prediction

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        # x = F.tanh(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flattens so can then be put in fully connected network
        # print(x.shape)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)  # No activation (MSELoss uses raw output). Softtmax didnt work, gave constant output, but I could see it working
        return x


data_set = Data_Set()

data_loader = DataLoader(data_set, batch_size=5, shuffle=True)


model = Net()
criterion = nn.MSELoss()
optimiser = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(2):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(data_loader, 0):
        optimiser.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()

        running_loss += loss.item()

        if i % 20 == 19:  # Print every 20 mini-batches
            print(f"Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss / 20:.4f}")
            running_loss = 0.0


# change to your path
torch.save(model.state_dict(), r"/Users/xavierparker/Desktop/Bot Stack/battleships/AI model/trained.pth")

print(outputs[0].shape)
print(inputs[0])
print(outputs[0])
print(labels[0])
print("Finished Training")

