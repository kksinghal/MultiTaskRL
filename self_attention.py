import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class self_attention(nn.Module):

    def __init__(self, in_channels, img_width, img_height, retention_time):
        super().__init__()

        self.in_channels = in_channels
        self.img_width = img_width
        self.img_height = img_height
        self.retention_time = retention_time

        self.query_conv =  nn.Conv3d(in_channels, int(in_channels/2), kernel_size=1)
        self.key_conv =  nn.Conv3d(in_channels, int(in_channels/2), kernel_size=1)
        self.value_conv =  nn.Conv3d(in_channels, int(in_channels/2), kernel_size=1)

        self.out_conv = nn.Conv3d(int(in_channels/2), in_channels, kernel_size=1, bias=False)

        self.softmax = torch.nn.Softmax(dim=2)

        self.task_query = torch.rand((in_channels, retention_time, img_width, img_height))
        self.task_key = torch.rand((in_channels, retention_time, img_width, img_height))
        self.task_value = torch.rand((in_channels, retention_time, img_width, img_height))

        self.positional_encoding = torch.rand((in_channels, retention_time, img_height, img_width))
        

    def forward(self, X):

        query = (self.query_conv(X+self.positional_encoding) + self.task_query).reshape(-1, int(self.in_channels/2), self.img_width*self.img_height*self.retention_time).permute(0,2,1)
        key = (self.key_conv(X+self.positional_encoding) + self.task_key).reshape(-1, int(self.in_channels/2), self.img_width*self.img_height*self.retention_time)

        similarity = torch.matmul(query, key)
        softmax = self.softmax(similarity)

        value = (self.value_conv(X +self.positional_encoding) + self.task_value).reshape(-1, int(self.in_channels/2), self.img_width*self.img_height*self.retention_time).permute(0,2,1)
        
        out = torch.matmul(similarity, value).reshape(-1, int(self.in_channels/2), self.retention_time, self.img_height, self.img_width)

        out = self.out_conv(out)
        out = out + X

        return out
        
