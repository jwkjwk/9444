# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.linear = nn.Linear(28*28,10)
        self.log_softmax = nn.LogSoftmax()
        # INSERT CODE HERE

    def forward(self, x):
        x1 = x.view(x.shape[0],-1)
        x2 = self.linear(x1)
        x3 = self.log_softmax(x2)

        return x3 # CHANGE CODE HERE


class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        # INSERT CODE HERE
        self.linear = nn.Linear(28*28,110)
        self.linear2 = nn.Linear(110,10)
        self.th = nn.Tanh()
        self.log_softmax = nn.LogSoftmax()


    def forward(self, x):
    # CHANGE CODE HERE
        x1 = x.view(x.shape[0],-1)
        x2 = self.linear(x1)
        x3 = self.th(x2)
        x4 = self.linear2(x3)
        x5 = self.log_softmax(x4)
        return x5

    
class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE
        self.convo1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 5)
        self.convo2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5)
        self.linear1 = nn.Linear(12800,100)
        self.linear2 = nn.Linear(100,10)
        self.log_softmax = nn.LogSoftmax()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size = 5)

    def forward(self, x):
        x1 = self.convo1(x)
        x2 = self.relu(x1)
        x3 = self.convo2(x2)
        x4 = self.relu(x3)
        x5 = x4.view(x4.shape[0],-1)
        x6 = self.linear1(x5)
        x7 = self.relu(x6)
        x8 = self.linear2(x7)
        x9 = self.log_softmax(x8)
        return x9 # CHANGE CODE HERE

