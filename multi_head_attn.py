import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class multi_head_attn(nn.Module):
    def __init__(self, n_heads, n_features, img_width, img_height, downsampled_channels):
        super().__init__()
        self.WQ_t_minus_1 = torch.rand(n_heads, n_features, 1, 1).to(device)
        self.WQ_x = torch.rand(1, n_features, 1, 1).to(device)
        self.BQ = torch.rand(n_heads, n_features, 1, 1).to(device)
        
        self.WK_x = torch.rand(n_features, 1, 1).to(device)
        self.BK = torch.rand(n_features, 1, 1).to(device)
        
        self.prev_Q = torch.zeros(n_heads, n_features, img_width, img_height).to(device)
        
        self.Vconv = nn.Conv2d(in_channels=n_features, out_channels=downsampled_channels, kernel_size=3, padding=1).to(device)
        

    def forward(self, X, WQ_task, BQ_task, WK_task, BK_task, WV_task, BV_task):
        
        reshaped_X = X.reshape(1, *X.shape) #Add dimension at the beginning
        Q = WQ_task * ( self.WQ_t_minus_1*self.prev_Q + self.WQ_x*reshaped_X + self.BQ) + BQ_task

        K = WK_task * ( self.WK_x*X + self.BK ) + BK_task
        
        reshaped_Q = Q.reshape(*Q.shape, 1, 1)
        K = K.reshape(*K.shape[:1], 1, 1, *K.shape[1:])

        softmax = torch.nn.Softmax(dim = 2)

        similarity = torch.sum(reshaped_Q*K, dim=[0,1])

        attention = softmax(similarity.reshape(*similarity.shape[:2], -1))
        attention = attention.reshape(*attention.shape[:2], 1, attention.shape[-1])

        V = WV_task * self.Vconv(reshaped_X) + BV_task

        V = V.reshape(1, *V.shape[:2], -1)

        out = torch.sum(attention * V, dim = 3).permute([2,0,1])
        
        self.prev_Q = Q.detach()
        
        return out
        
