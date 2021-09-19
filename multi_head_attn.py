import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class multi_head_attn(nn.Module):
    def __init__(self, n_heads, n_features, img_width, img_height, downsampled_channels):
        super().__init__()

        """
        Initialisation:
        self.WQ_t_minus_1 = torch.rand(n_heads, n_features, 1, 1).to(device)
        self.WQ_x = torch.rand(1, n_features, 1, 1).to(device)
        self.BQ = torch.rand(n_heads, n_features, 1, 1).to(device)
        self.WK_x = torch.rand(n_features, 1, 1).to(device)
        self.BK = torch.rand(n_features, 1, 1).to(device)
        self.Vconv = nn.Conv2d(in_channels=n_features, out_channels=downsampled_channels, kernel_size=3, padding=1).to(device)
        """

        self.relu = nn.ReLU()

        parameters = torch.load("./parameters/multi_head_attn", map_location=device)
        self.WQ_t_minus_1 = parameters["WQ_t_minus_1"].to(device)
        self.WQ_x = parameters["WQ_x"].to(device)
        self.BQ = parameters["BQ"].to(device)
        self.WK_x = parameters["WK_x"].to(device)
        self.BK = parameters["BK"].to(device)
        self.Vconv = parameters["Vconv"].to(device)

        self.prev_Q = torch.zeros(n_heads, n_features, img_width, img_height).to(device)


    def forward(self, X, WQ_task, BQ_task, WK_task, BK_task, WV_task, BV_task):
        
        reshaped_X = X.reshape(1, *X.shape) #Add dimension at the beginning
        Q = WQ_task * ( self.relu( self.WQ_t_minus_1*self.prev_Q + self.WQ_x*reshaped_X + self.BQ) ) + BQ_task

        K = WK_task * ( self.relu( self.WK_x*X + self.BK ) ) + BK_task
        
        reshaped_Q = Q.reshape(*Q.shape, 1, 1)
        K = K.reshape(*K.shape[:1], 1, 1, *K.shape[1:])

        softmax = torch.nn.Softmax(dim = 2)

        similarity = torch.sum(reshaped_Q*K, dim=[0,1])

        attention = softmax(similarity.reshape(*similarity.shape[:2], -1)).float()
        attention = attention.reshape(*attention.shape[:2], 1, attention.shape[-1]).float()

        V = WV_task * self.Vconv(reshaped_X) + BV_task

        V = V.reshape(1, *V.shape[:2], -1)

        out = torch.sum(attention * V, dim = 3).permute([2,0,1])
        
        self.prev_Q = Q.detach().clone().to(device)
        
        return out
        
    def save_parameters(self):
        parameters = {}
        parameters["WQ_t_minus_1"] = self.WQ_t_minus_1
        parameters["WQ_x"] = self.WQ_x
        parameters["BQ"] = self.BQ
        parameters["WK_x"] = self.WK_x
        parameters["BK"] = self.BK
        parameters["Vconv"] = self.Vconv
        torch.save(parameters, "./parameters/multi_head_attn")