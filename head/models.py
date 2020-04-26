import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import ipdb


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def shortest_dist(dist_mat):
    """Parallel version.
    Args:
    dist_mat: pytorch Variable, available shape:
      1) [m, n]
      2) [m, n, N], N is batch size
      3) [m, n, *], * can be arbitrary additional dimensions
    Returns:
    dist: three cases corresponding to `dist_mat`:
      1) scalar
      2) pytorch Variable, with shape [N]
      3) pytorch Variable, with shape [*]
    """
    m, n = dist_mat.size()[:2]
    # Just offering some reference for accessing intermediate distance.
    dist = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if (i == 0) and (j == 0):
                dist[i][j] = dist_mat[i, j]
            elif (i == 0) and (j > 0):
                dist[i][j] = dist[i][j - 1] + dist_mat[i, j]
            elif (i > 0) and (j == 0):
                dist[i][j] = dist[i - 1][j] + dist_mat[i, j]
            else:
                dist[i][j] = torch.min(dist[i - 1][j], dist[i][j - 1]) + dist_mat[i, j]
    dist = dist[-1][-1]
    return dist

def local_dist(x, y):
    """
    Args:
    x: pytorch Variable, with shape [M, m, d]
    y: pytorch Variable, with shape [N, n, d]
    Returns:
    dist: pytorch Variable, with shape [M, N]
    """
    M, m, d = x.size()
    N, n, d = y.size()
    x = x.contiguous().view(M * m, d)
    y = y.contiguous().view(N * n, d)
    # shape [M * m, N * n]
    dist_mat = euclidean_dist(x, y)
    dist_mat = (torch.exp(dist_mat) - 1.) / (torch.exp(dist_mat) + 1.)
    # shape [M * m, N * n] -> [M, m, N, n] -> [m, n, M, N]
    dist_mat = dist_mat.contiguous().view(M, m, N, n).permute(1, 3, 0, 2)
    # shape [M, N]
    dist_mat = shortest_dist(dist_mat)
    return dist_mat

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
#         self.relu2 = nn.Linear()
#         self.max_pool = nn.MaxPool2d(kernel_size=(4, 1), stride=4, padding=0)
        self.lin = nn.Linear(2048, 1)
        self.drop = nn.Dropout(.4)
    
        self.sig = nn.Sigmoid()
        
    def forward(self, y1, y2):
        par1 = y1*y2
        par2 = y1+y2
        par3 = torch.abs(y1 - y2)
        par4 = par3**2
#         ipdb.set_trace()
        x = torch.cat([par1, par2, par3, par4], 1)
#         ipdb.set_trace()
        x = x.reshape(-1, 1, 4, 2048)
        x = self.relu1(self.conv1(x))
#         ipdb.set_trace()
        x = x.reshape(-1, 1, 2048, 128)
        x = self.conv2(x)
#         ipdb.set_trace()
#         x = self.avg_pool(x)
#         ipdb.set_trace()
        x = x.view(x.size(0), -1) 
        x = self.lin(self.drop(x))              
        y = self.sig(x)
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