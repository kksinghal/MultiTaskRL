import torch
from torch import nn
import torchvision
from torchvision import transforms

from critic import critic
from actor import actor

from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

actor_learning_rate=1e-4 
critic_learning_rate=1e-3
gamma=0.99
tau=1e-2
max_buffer_size=50000

class Agent():
    def __init__(self, retention_time): 

        self.retention_time = retention_time
        
        self.critic = critic(retention_time)
        self.critic_target = critic(retention_time)

        self.actor = actor(retention_time)
        self.actor_target = actor(retention_time)
        
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Training
        self.replay_buffer = ReplayBuffer(max_buffer_size)        
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        self.memory = None

    
    def add_to_memory(self, state):
        self.memory = torch.cat((self.memory[1:], state))


    def get_action(self):
        state = self.transform(self.memory)
        action = self.actor(self.memory)
        action = action.detach().numpy()
        return action
    

    def update(self, batch_size):
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
    
        # Critic loss        
        Qvals = self.critic(states, actions)
        next_actions = self.actor_target(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        policy_loss = -self.critic(states, self.actor(states)).mean()
        
        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()

        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    
        