import torch
from torch import nn
import torchvision

from multi_head_attn import multi_head_attn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent(nn.Module):
    def __init__(self, n_heads): #n_heads for multi head attention
        super().__init__()
        
        self.n_heads = n_heads
        
        resnet18 = torchvision.models.resnet18(pretrained=True)
        self.resnet_preprocessing_model = nn.Sequential(*list(resnet18.children())[:-3])
        for param in self.resnet_preprocessing_model.parameters():
            param.requires_grad = False
        
        self.memory = {}
        
        self.attention_model = multi_head_attn(self.n_heads, 256, 16, 16, 64)
        
        self.actor_fc = nn.Sequential(
            nn.Linear(64*16*16, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 4) 
        )#Output is forward_force_mean, forward_force_std, angular_velocity_mean, angular_velocity_std
        
        self.critic_fc = nn.Sequential(
            nn.Linear(64*16*16 + 4, 128), #4 for the actions
            nn.ReLU(),
            nn.Linear(128, 32),#Output value for the action in given state
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        
    
    def forward(self, X, task):
        reshaped_X = X.reshape(1, *X.shape)
        out = self.resnet_preprocessing_model(reshaped_X).squeeze()
        
        out = self.attention_model(out, *self.memory[task].values())
        
        out = torch.flatten(out)
        action_dist = self.actor_fc(out)
        
        action_dist[[1,3]] = torch.abs(action_dist[[1,3]])
        value = self.critic_fc(torch.cat((out, action_dist)))
        return action_dist, value
    
        
    def get_brain_parameters(self):
        return list(self.attention_model.parameters()) + list(self.actor_fc.parameters()) + list(self.critic_fc.parameters())
            

    def get_task_memory_parameters(self, task):
        if task not in self.memory.keys():
            self.memory[task] = {
                "WQ": torch.rand((self.n_heads,256,1,1)),
                "BQ": torch.rand((self.n_heads,256,1,1)),
                "WK": torch.rand((256,1,1)),
                "BK": torch.rand((256,1,1)),
                "WV": torch.rand((64,1,1)),
                "BV": torch.rand((64,1,1))
            }
            for key, value in self.memory[task].items():
                self.memory[task][key] = self.memory[task][key].to(device)
        return self.memory[task].values()
    
