import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) # padding = 1 we add a border with zeros, otherwise it will be 26*26 
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 32 -> 64 -> 128 -> 256 ...
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 14)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))  # 28x28
        x = self.pool(x)              # 14x14
        x = self.relu(self.conv2(x))
        x = self.pool(x)              # 7x7

        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

