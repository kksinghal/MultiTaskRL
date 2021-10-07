import torch
from torch import nn
import torchvision
from torchvision import transforms

from critic import critic
from actor import actor

from utils import *

from pathlib import Path

device1 = torch.device("cuda:0") # critic
device2 = torch.device("cuda:0") # actor

max_buffer_size = 10000
actor_learning_rate=1e-5
critic_learning_rate=1e-4

class Agent():
    def __init__(self, retention_time): 

        self.retention_time = retention_time
        
        self.critic = critic(retention_time).to(device1)
        critic_file = Path("./parameters/critic")
        if critic_file.is_file():
            self.critic.load_state_dict(torch.load("./parameters/critic"))
        self.critic_target = critic(retention_time).to(device1)

        self.actor = actor(retention_time).to(device2)
        actor_file = Path("./parameters/actor")
        if actor_file.is_file():
            self.actor.load_state_dict(torch.load("./parameters/actor"))
        self.actor_target = actor(retention_time).to(device2)
        
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Training
        self.replay_buffer = ReplayBuffer(max_buffer_size, retention_time)        
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        self.memory = None

        self.noise = OUActionNoise(np.zeros((1,3)))

    
    def add_to_memory(self, obs): # memory.shape = 1, filters, retention_time, img_size, img_size
        obs = obs.reshape(1, obs.shape[0], 1, *obs.shape[1:])
        self.memory = torch.cat((obs, self.memory[:, :, :-1, :, :]), dim=2)


    def get_action(self):
        shape = self.memory.shape
        memory = self.memory.reshape(shape[0]*shape[2], 3, shape[-2], shape[-1])

        x = self.transform(memory).reshape(*shape).float().to(device2)

        action = self.actor(x).cpu()
        #noise = torch.tensor(self.noise(), dtype=action.dtype)

        #action += noise

        return action.detach().numpy() 
    
