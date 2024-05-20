import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN, self).__init__()
        self.layer1 = nn.Linear(in_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, out_dim)

    def forward(self, x):

        activation1 = F.relu(self.layer1(x))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)
        return output
