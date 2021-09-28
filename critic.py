import torch
from torch import nn

from ResNet import ResNet

class critic(nn.Module):
    def __init__(self, retention_time):
        super().__init__()
        
        self.resnet = ResNet(in_channels=3, img_size=256, retention_time=retention_time)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(2048*retention_time*4*4 + 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, X, actions):
        y = self.resnet(X)
        y = self.flatten(y)

        y = torch.cat((y, actions), dim=1)
        out = self.fc(y)

        return out