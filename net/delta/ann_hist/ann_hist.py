import torch
import torch.nn as nn
import torch.nn.functional as F


class ANN_HIST(nn.Module):
    def __init__(self, len_op=10):
        super(ANN_HIST, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(201, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, len_op)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        return F.log_softmax(x, dim=1)


class ANN_HIST1(nn.Module):
    name = 'delta_ann1'
    alpha = 0.01
    momentum = 0.5
    epoch = 10
    batch = 64
    loss = 'cross entropy'

    def __init__(self, len_op=10):
        super(ANN_HIST1, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(201, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, len_op)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return F.log_softmax(x, dim=1)