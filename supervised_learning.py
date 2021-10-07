import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Agent import Agent
from utils import *
import random

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('tensorboard/')

def set_seeds(seed):
    torch.manual_seed(seed)  # Sets seed for PyTorch RNG
    torch.cuda.manual_seed_all(seed)  # Sets seeds of GPU RNG
    np.random.seed(seed=seed)  # Set seed for NumPy RNG
    random.seed(seed)  # Set seed for random RNG

set_seeds(1)

retention_time = 3
agent = Agent(retention_time)
batch_size = 32 #Actual batch_size = 32*4
rewards = []
avg_rewards = []
epsilon = 0.1
gamma=0.99
tau=1e-2

T_MAX = 300


buffer_data = pd.read_csv("./data/buffer_data.csv")
for index, row in buffer_data.iterrows():
    state = torch.load("./data/states/"+str(row["id"])+".pt")
    new_state = torch.load("./data/next_states/"+str(row["id"])+".pt")
    action = [[row["forward_force"], row["right_force"], row["angular"]]]
    agent.replay_buffer.push(state, torch.tensor(action), row["reward"], new_state, row["done"])

agent.replay_buffer.sampling_ids = np.load("./data/sampling_ids.npy")


class my_dataset(Dataset):
    def __init__(self):
        self.buffer = agent.replay_buffer.buffer
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    def __len__(self):
        return len(agent.replay_buffer.sampling_ids)

    def random_flip(self, current_state_memory, action, next_state_memory):
        if random.uniform(0, 1) < 0.5:
            current_state_memory = torch.flip(current_state_memory, dims=[3])
            next_state_memory = torch.flip(next_state_memory, dims=[3])
            action = torch.tensor([action[0], -1*action[1], -1*action[2]])

        return current_state_memory, action, next_state_memory

    def __getitem__(self, idx):

        state, action, reward, next_state, done = self.buffer[idx]
        reward, done = torch.tensor(reward), torch.tensor(done)
        action = action[0]
        current_state_memory = state.reshape(state.shape[0], 1, *state.shape[1:])
        for _id in range(idx-1, idx-retention_time, -1):
            old_state = self.buffer[_id][0]
            old_state = old_state.reshape(old_state.shape[0], 1, *old_state.shape[1:])

            current_state_memory = torch.cat((current_state_memory, old_state), dim=1)
        
        next_state = next_state.reshape(next_state.shape[0], 1, *next_state.shape[1:])
        next_state_memory = torch.cat((next_state, current_state_memory[:,:-1]), dim=1)

        current_state_memory = self.transform(current_state_memory.permute(1,0,2,3)).permute(1,0,2,3)
        next_state_memory = self.transform(next_state_memory.permute(1,0,2,3)).permute(1,0,2,3)

        #Augmentation
        current_state_memory, action, next_state_memory = self.random_flip(current_state_memory, action, next_state_memory)
        return current_state_memory, action, reward, next_state_memory, done


dataset = my_dataset()

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)


device1 = torch.device("cuda:0") # critic
device2 = torch.device("cuda:1") # actor


def take_step():
    agent.critic_optimizer.step()
    agent.actor_optimizer.step()

    agent.critic_optimizer.zero_grad()
    agent.actor_optimizer.zero_grad()

    # update target networks 
    for target_param, param in zip(agent.actor_target.parameters(), agent.actor.parameters()):
        target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))
    
    for target_param, param in zip(agent.critic_target.parameters(), agent.critic.parameters()):
        target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))


def train_loop(epoch):
    size = len(train_dataloader.dataset)
    total_critic_loss, total_actor_loss = 0, 0
    for batch, (states, actions, rewards, next_states, done_batch) in enumerate(train_dataloader):
        states_device1 = states.to(device1)
        states_device2 = states.to(device2)

        actions_device1 = actions.to(device1)
        
        next_states_device1 = next_states.to(device1)
        next_states_device2 = next_states.to(device2)
        

        agent.actor_target.eval()
        agent.critic_target.eval()
        agent.critic.eval()
        # Critic loss
        Qvals = agent.critic(states_device1, actions_device1).squeeze()
        next_actions = agent.actor_target(next_states_device2).to(device1)
        next_Q = agent.critic_target(next_states_device1, next_actions.detach()).squeeze().cpu()
        Qprime = (rewards + gamma * next_Q * done_batch).to(device1)

        agent.critic.train()
        critic_loss = agent.critic_criterion(Qvals.float(), Qprime.float())
        critic_loss.backward(retain_graph=True) 

        agent.critic.eval()
        
        # Actor loss
        actions_ = agent.actor(states_device2).to(device1)
        agent.actor.train()
        policy_loss = -agent.critic(states_device1, actions_).mean()
        policy_loss.backward()

        if (batch+1)%batch_size == 0:
            take_step()
    
        total_actor_loss += policy_loss.detach()
        total_critic_loss += critic_loss.detach()

    take_step()
    print(f"Epoch {epoch}; actor loss: {total_actor_loss:>7f}, critic loss: {total_critic_loss:>7f}")

    torch.save(agent.actor.state_dict(), "./parameters/actor")
    torch.save(agent.critic.state_dict(), "./parameters/critic")
    return total_actor_loss, total_critic_loss


def eval_loop():
    size = len(test_dataloader.dataset)
    total_critic_loss, total_actor_loss = 0, 0
    for batch, (states, actions, rewards, next_states, done_batch) in enumerate(test_dataloader):
        states_device1 = states.to(device1)
        states_device2 = states.to(device2)

        actions_device1 = actions.to(device1)
        
        next_states_device1 = next_states.to(device1)
        next_states_device2 = next_states.to(device2)

        # Critic loss
        Qvals = agent.critic(states_device1, actions_device1).squeeze()
        next_actions = agent.actor_target(next_states_device2).to(device1)
        next_Q = agent.critic_target(next_states_device1, next_actions.detach()).squeeze().cpu()
        Qprime = (rewards + gamma * next_Q * done_batch).to(device1)

        critic_loss = agent.critic_criterion(Qvals.float(), Qprime.float()).detach().cpu()

        # Actor loss
        policy_loss = -agent.critic(states_device1, agent.actor(states_device2).to(device1)).mean().detach().cpu()

        total_actor_loss += policy_loss
        total_critic_loss += critic_loss

    return total_actor_loss, total_critic_loss

epochs = 400
for epoch in range(15, epochs):
    train_actor_loss, train_critic_loss = train_loop(epoch)

    with torch.no_grad():
        agent.critic.eval()
        agent.actor.eval()
        agent.critic_target.eval() 
        agent.actor_target.eval()
        test_actor_loss, test_critic_loss = eval_loop()
        agent.critic.train()
        agent.actor.train()
        agent.critic_target.train() 
        agent.actor_target.train()

    writer.add_scalars('Critic Loss', {
        "Train": train_critic_loss/train_size,
        "Eval": test_critic_loss/test_size
        }, epoch)

    writer.add_scalars('Actor Loss', {
        "Train": train_actor_loss/train_size,
        "Eval": test_actor_loss/test_size
        }, epoch)

    if (epoch+1)%20 == 0:
        for target_param, param in zip(agent.actor_target.parameters(), agent.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(agent.critic_target.parameters(), agent.critic.parameters()):
            target_param.data.copy_(param.data)


