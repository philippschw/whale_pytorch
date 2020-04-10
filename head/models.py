import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import ipdb
    
class HeadWhaleModel(nn.Module):
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4,1), padding=(0, 0), dilation=(1, 1))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,128))
        self.ap2d = nn.AdaptiveAvgPool2d(1)
        self.out_act = nn.Sigmoid()
        
    def forward(self, y1, y2):
        par1 = y1*y2
        par2 = y1+y2
        par3 = torch.abs(y1 - y2)
        par4 = par3**2
        x = torch.cat([par1, par2, par3, par4], 1)
        x = x.reshape(-1, 1, 4, 2048)
        x = self.relu1(self.conv1(x))
#         ipdb.set_trace()
        x = x.reshape(-1, 1, 2048, 128)
        x = self.ap2d(self.conv2(x))
#         ipdb.set_trace()
        x = x.view(-1, self.num_flat_features(x))
        y = self.out_act(x)
        return y
    
    def getLoss(self, output, labels):
        criterion = nn.BCELoss()
        self.loss = criterion(output, labels)
        
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