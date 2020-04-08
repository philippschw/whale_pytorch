import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

    
class HeadWhaleModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.dout0 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(4096, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu1 = nn.ReLU()
        self.dout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.dout2 = nn.Dropout(0.3)
        self.out = nn.Linear(256, 1)
        self.out_act = nn.Sigmoid()
        
    def forward(self, input_):
        a1 = self.bn1(self.fc1(self.dout0(input_)))
        h1 = self.relu1(a1)
        dout = self.dout1(h1)
        a2 = self.bn2(self.fc2(dout))
        h2 = self.dout2(self.relu2(a2))
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y
    
    def getLoss(self, results, labels):
        criterion = nn.BCELoss()
        self.loss = criterion(results, labels)
        
    def load_pretrain(self, pretrain_file, skip=[]):
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()

        keys = list(state_dict.keys())
        for key in keys:
            if any(s in key for s in skip): continue
            try:
                state_dict[key] = pretrain_state_dict[key]
            except:
                print(key)

        self.load_state_dict(state_dict)