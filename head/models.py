import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

def threashold_contrastive_loss(input1, input2, m):
    """dist < m --> 1 else 0"""
    diff = input1 - input2
    dist_sq = torch.sum(torch.pow(diff, 2), 1)
    dist = torch.sqrt(dist_sq)
    threashold = dist.clone()
    threashold.data.fill_(m)
    return (dist < threashold).float().view(-1, 1)
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def check_type_forward(self, in_types):
        assert len(in_types) == 3

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, x0, x1, y):
        self.check_type_forward((x0, x1, y))

        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss
    
class HeadWhaleModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        # self.dout0 = nn.Dropout(0.5)
        # self.fc1 = nn.Linear(4096, 1024)
        # self.bn1 = nn.BatchNorm1d(1024)
        # self.relu1 = nn.ReLU()
        # self.dout1 = nn.Dropout(0.5)
        # self.fc2 = nn.Linear(1024, 256)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.relu2 = nn.ReLU()
        # self.dout2 = nn.Dropout(0.3)
        # self.out = nn.Linear(256, 1)
        # self.out_act = nn.Sigmoid()
        
    def forward(self, y1, y2):
        # a1 = self.bn1(self.fc1(self.dout0(input_)))
        # h1 = self.relu1(a1)
        # dout = self.dout1(h1)
        # a2 = self.bn2(self.fc2(dout))
        # h2 = self.dout2(self.relu2(a2))
        # a3 = self.out(h2)
        # y = self.out_act(a3)
        return y1, y2
    
    def getLoss(self, output1, output2, labels):
        # criterion = nn.BCELoss()
        criterion = ContrastiveLoss(margin=1)
        self.loss = criterion(output1, output2, labels)
        
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