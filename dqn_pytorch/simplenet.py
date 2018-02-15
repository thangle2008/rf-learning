import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleANN(nn.Module):

    def __init__(self, input_size, output_size):
        super(SimpleANN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.head = nn.Linear(24, self.output_size)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x)
