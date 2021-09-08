import torch
from torch import nn

from multi_head_attn import multi_head_attn

class BrainClass(nn.Module):
    def __init__(self, n_heads): #n_heads for multi head attention
        super().__init__()
        
        self.n_heads = n_heads
        
        resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.resnet_preprocessing_model = nn.Sequential(*list(resnet18.children())[:-3])
        
        self.memory = {}
        
        self.attention_model = multi_head_attn(self.n_heads, 256, 16, 16, 64)
        
        self.actor_fc = nn.Sequential(
            nn.Linear(64*16*16, 128),
            nn.Linear(128, 32),
            nn.Linear(32, 4) 
        )#Output is forward_force_mean, forward_force_std, angular_velocity_mean, angular_velocity_std
        
        self.critic_fc = nn.Sequential(
            nn.Linear(64*16*16 + 4, 128), #4 for the actions
            nn.Linear(128, 32),#Output value for the action in given state
            nn.Linear(32, 1) 
        )
        
    
    def forward(self, X, task):
        out = self.resnet_preprocessing_model(X.reshape(1, *X.shape)).reshape(256, 16, 16)
        out = self.attention_model(out, *self.memory[task].values())
        
        out = out.reshape(-1)
        
        action = self.actor_fc(out)
        value = self.critic_fc(torch.cat(out, action))
        return action, value
        
        
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
        return self.memory[task].values()