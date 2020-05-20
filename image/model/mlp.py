import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(config['d_input'], 512)
        self.fc2 = nn.Linear(512, config['n_class'])
        self.bn1 = nn.BatchNorm1d(config['d_input'])
        self.bn2 = nn.BatchNorm1d(512)
        self.drops = nn.Dropout(0.1)

    def forward(self, x):
        x = self.bn1(x)
        x = self.drops(x)
        x = F.relu(self.fc1(x))
        x = self.bn2(x)
        x = self.drops(x)
        x = self.fc2(x)
        return x