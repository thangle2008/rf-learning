import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleANN(nn.Module):

    def __init__(self, in_features, num_actions):
        super(SimpleANN, self).__init__()
        self.num_actions = num_actions
        self.fc1 = nn.Linear(in_features, 24)
        self.fc2 = nn.Linear(24, 24)
        self.head = nn.Linear(24, num_actions)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x)


class AtariConvNet(nn.Module):

    def __init__(self, in_channels, num_actions):
        super(AtariConvNet, self).__init__()
        self.num_actions = num_actions
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, num_actions)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
