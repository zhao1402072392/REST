import torch
import torch.nn as nn


class CNN(nn.Module):
    # For cmnist only
    def __init__(self):
        super().__init__()
        self.last_dim = 256

        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        self.num_classes=10

        self.net = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.conv2,
            self.bn2,
            self.relu,
            self.conv3,
            self.bn3,
            self.relu,
            self.avgpool,
        )
        self.fc = nn.Linear(self.last_dim, self.num_classes)

    def forward(self, x):
        out = self.net(x)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


