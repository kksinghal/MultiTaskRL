import torch
from torch import nn
import torchvision
from torchvision import transforms

from critic import critic
from actor import actor

from utils import *

from pathlib import Path

device1 = torch.device("cuda:0") # critic
device2 = torch.device("cuda:1") # actor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

actor_learning_rate=1e-4
critic_learning_rate=1e-4
gamma=0.99
tau=1e-2
max_buffer_size = 10000

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

        x = self.transform(memory).reshape(*shape).float().to(device)

        action = self.actor(x).cpu()
        #noise = torch.tensor(self.noise(), dtype=action.dtype)

        #action += noise

        return action.detach().numpy() 
    

    def update(self, batch_size):

        states, actions, rewards, next_states, done_batch = self.replay_buffer.sample(batch_size, self.retention_time)
        states = torch.stack([x.float() for x in states]).to(device)
        actions = torch.stack([x.float() for x in actions]).to(device)
        rewards = torch.tensor(rewards).squeeze()
        done_batch = torch.tensor(done_batch)
        next_states = torch.stack([x.float() for x in next_states]).to(device)

        # Critic loss        
        Qvals = self.critic(states, actions).squeeze()
        next_actions = self.actor_target(next_states)
        next_Q = self.critic_target(next_states, next_actions.detach()).squeeze().cpu()

        Qprime = (rewards + gamma * next_Q * done_batch).to(device)

        critic_loss = self.critic_criterion(Qvals.float(), Qprime.float())

        # Actor loss
        policy_loss = -self.critic(states, self.actor(states)).mean()
        
        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True, inputs=list(self.actor.parameters()))
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward(inputs=list(self.critic.parameters())) 
        self.critic_optimizer.step()



        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))

    
        return policy_loss.detach().cpu(), critic_loss.detach().cpu()
        