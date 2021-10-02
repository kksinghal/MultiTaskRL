import torch
from torch import nn

from ResNet import ResNet

class actor(nn.Module):
    def __init__(self, retention_time):
        super().__init__()

        self.resnet = ResNet(in_channels=3, img_size=128, retention_time=retention_time)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(2048*3*2*2, 1024),
            nn.ReLU(inplace=False),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=False),
            nn.Linear(512, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, 64),
            nn.ReLU(inplace=False),
            nn.Linear(64, 3),
            nn.Tanh()
        )

    
    def forward(self, X):
        y = self.resnet(X)

        y1 = self.flatten(y)

        out = self.fc(y1) * 2 # Clip actions between -2 and 2

        return out