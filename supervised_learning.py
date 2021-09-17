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
    
batch_size = 8
dataloader = DataLoader(EpisodeSelector, shuffle=True, batch_size=batch_size)

lr = 1e-3

brain_optimizer = torch.optim.Adam(agent.get_brain_parameters(), lr=lr*1e-3, 
                    betas=(0.92, 0.999))
loss_fn = nn.MSELoss()

def train_loop():
    size = len(dataloader)
    for batch, (task, episode_idx, label_df) in enumerate(dataloader):

        #agent.attention_model.prev_Q = torch.zeros(16, 256, 16, 16).to(device) 

        task_memory_optimizer = torch.optim.Adam(agent.get_task_memory_parameters(task), lr=lr, 
                            betas=(0.92, 0.999))

        loss = 0
        for index, row in label_df.iterrows():
            observation = torchvision.io.read_image("./data/"+task+"/"+str(episode_idx)+"/"+str(int(row["id"]))+".png").float()

            pred_action_dist, pred_value = agent(observation, task)

            action_dist = torch.tensor([row["forward_force_mean"], 5, row["angular_velocity_mean"], 5])
            value = torch.tensor([row["value"]])

            loss += loss_fn(pred_action_dist.float(), action_dist.float())
            loss += loss_fn(pred_value.float(), value.float())

        writer.add_scalar('training loss',
            batch / 1000,
            episode_idx)

        brain_optimizer.zero_grad()
        task_memory_optimizer.zero_grad()

        loss.backward()

        brain_optimizer.step()
        task_memory_optimizer.step()

        #if batch % 100 == 0:
        loss, current = loss.item(), batch
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


train_loop()




