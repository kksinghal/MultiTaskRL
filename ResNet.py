from numpy import NaN
import torch
from torch import nn
import torchvision
import sys

from self_attention import self_attention


class bottle_neck(nn.Module):
    def __init__(self, in_channels, img_size, retention_time):
        super().__init__()
        
        self.retention_time = retention_time
        self.img_size = img_size

        self.attn_1 = self_attention(in_channels, img_size, img_size, retention_time)
        self.batch_norm1 = nn.BatchNorm3d(in_channels)

        self.sequential = nn.Sequential(
            self.attn_1,
            self.batch_norm1,
        )
            
        
    def forward(self, X):
        y = self.sequential(X)

        return y
        
        
class downsample(nn.Module):
    def __init__(self, in_channels, retention_time, img_size):
        super().__init__()
        self.out_channels = in_channels*2
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels*2, kernel_size=3, padding=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        self.img_size = img_size
        self.retention_time = retention_time

    
    def forward(self, X):
        shape = X.shape
        out = self.conv1(X).reshape(shape[0], -1, shape[-2], shape[-1])
        out = self.avg_pool(out).reshape(shape[0], self.out_channels, self.retention_time, int(self.img_size/2), int(self.img_size/2))
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, img_size, retention_time):
        super(ResNet, self).__init__()
        
        self.stem = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )

        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        self.flatten = nn.Flatten()

        self.blocks = nn.ModuleList([]) 
        for i in range(5):
            self.blocks.append(self.make_layer(64*(2**i), int(img_size/(2**(i+1))), retention_time, n_blocks= 2))


    def forward(self, X):
        X = self.stem(X)

        shape = X.shape
        X = X.reshape(shape[0], -1, shape[-2], shape[-1])
        X = self.maxPool(X).reshape(*shape[:3], int(shape[3]/2), int(shape[4]/2))

        for block in self.blocks:
            X = block(X)

        return X
        

    def make_layer(self, in_channels, img_size, retention_time, n_blocks):
        layers= []
        layers.append(downsample(in_channels, retention_time, img_size))
        for i in range(n_blocks):
            layers.append(bottle_neck(in_channels*2, int(img_size/2), retention_time))
        
        
        return nn.Sequential(*layers)