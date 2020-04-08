import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

class HeadWhaleModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4096, 1024)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, 256)
        self.relu2 = nn.ReLU(1)
        self.out = nn.Linear(256, 1)
        self.out_act = nn.Sigmoid()
        
    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        a2 = self.fc2(dout)
        h2 = self.relu2(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y
    
    def getLoss(self, results, labels):
        criterion = nn.BCELoss()
        self.loss = criterion(results, labels)