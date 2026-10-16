import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(32*49, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.flatten(1)
        return F.relu(self.fc1(x))

    def forward(self, x):
        x = self.features(x)
        return self.fc2(x)


# 1 28 28
# 16 28 28
# 16 14 14 
# 32 14 14
# 32 7 7
