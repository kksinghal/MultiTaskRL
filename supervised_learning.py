import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import datasets, transforms

import numpy as np
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from Agent import Agent
agent = Agent(n_heads=16).to(device)

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('tensorboard/loss')


class EpisodeSelector(Dataset):
    def __init__(self):
        self.metadata = pd.read_csv("./data/metadata.csv")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        task = self.metadata.iloc[idx]["task"]
        episode_idx = self.metadata.iloc[idx,1]
        
        output = pd.read_csv("./data/"+task+"/"+str(episode_idx)+"/output.csv")
        return task, episode_idx, output
    

episode_selector = EpisodeSelector()
lr = 1e-2

brain_optimizer = torch.optim.Adam(agent.get_brain_parameters(), lr=lr,#*1e-2, 
                    betas=(0.92, 0.999))
loss_fn = nn.MSELoss()

def train_loop(epoch):
    size = len(episode_selector)
    for batch, (task, episode_idx, label_df) in enumerate(episode_selector):

        agent.attention_model.prev_Q = torch.zeros(16, 256, 16, 16).to(device) 

        task_memory_optimizer = torch.optim.Adam(agent.get_task_memory_parameters(task), lr=lr, 
                            betas=(0.92, 0.999))

        loss = 0
        loss_action = 0
        loss_value = 0
        for index, row in label_df.iterrows():
            observation = torchvision.io.read_image("./data/"+task+"/"+str(episode_idx)+"/"+str(int(row["id"]))+".png").float().to(device)

            pred_action_dist, pred_value = agent(observation, task)
            print(pred_action_dist)
            action_dist = torch.tensor([row["forward_force_mean"], 5, row["right_force_mean"], 5, row["angular_velocity_mean"], 5]).to(device)
            value = torch.tensor([row["value"]]).to(device)

            loss += loss_fn(pred_action_dist.float(), action_dist.float())
            loss += loss_fn(pred_value.float(), value.float())

            loss_action += loss_fn(pred_action_dist.float(), action_dist.float())
            loss_value += loss_fn(pred_value.float(), value.float())

        writer.add_scalar('action loss',
            loss_action / 6000,
            batch + size*epoch)
        writer.add_scalar('value loss',
            loss_value / 1000,
            batch + size*epoch)

        brain_optimizer.zero_grad()
        task_memory_optimizer.zero_grad()

        loss.backward()

        brain_optimizer.step()
        task_memory_optimizer.step()

        #if batch % 100 == 0:
        loss, current = loss.item(), batch
        print(f"loss: {loss:>7f}  [Epoch: {epoch}; {current:>5d}/{size:>5d}]")

epochs = 310
for epoch in range(300, epochs):
    train_loop(epoch)
    agent.save_parameters()


