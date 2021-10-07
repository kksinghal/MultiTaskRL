import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Agent import Agent
from utils import *
import random
import keyboard
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms


from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('tensorboard/')

env_path = "./Scenes/PushBlock_Win_Small/UnityEnvironment"
env = UnityEnvironment(file_name=env_path,  seed=1, side_channels=[])

def set_seeds(seed):
    torch.manual_seed(seed)  # Sets seed for PyTorch RNG
    torch.cuda.manual_seed_all(seed)  # Sets seeds of GPU RNG
    np.random.seed(seed=seed)  # Set seed for NumPy RNG
    random.seed(seed)  # Set seed for random RNG

set_seeds(1)

retention_time = 3
agent = Agent(retention_time)
batch_size = 5
rewards = []
avg_rewards = []
epsilon = 0.1

gamma=0.99
tau=1e-2

T_MAX = 300

device1 = torch.device("cuda:0") # critic
device2 = torch.device("cuda:0") # actor

def get_input():
    forward_force_mean = 0 
    right_force_mean = 0 
    angular_velocity_mean = 0
    try:
        if keyboard.is_pressed('w'):
            forward_force_mean = 1
        if keyboard.is_pressed('f'):
            forward_force_mean = -1
        if keyboard.is_pressed('s'):
            forward_force_mean = -1
        
        if keyboard.is_pressed('d'):
            right_force_mean = 1
        if keyboard.is_pressed('a'):
            right_force_mean = -1

        if keyboard.is_pressed('m'):
            angular_velocity_mean = 1
        if keyboard.is_pressed('n'):
            angular_velocity_mean = -1    
        
    except:
        print("except")
    return np.array([[forward_force_mean, right_force_mean, angular_velocity_mean]])*2



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
        idx = agent.replay_buffer.sampling_ids[idx]
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


def train_loop():
    size = len(train_dataloader.dataset)
    total_critic_loss, total_actor_loss = 0, 0
    for batch, (states, actions, rewards, next_states, done_batch) in enumerate(train_dataloader):
        states_device1 = states.to(device1).float()
        states_device2 = states.to(device2).float()

        actions_device1 = actions.to(device1).float()
        
        next_states_device1 = next_states.to(device1).float()
        next_states_device2 = next_states.to(device2).float()
        

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

        current = batch * len(done_batch)
        print(f"[{current:>5d}/{size:>5d}]", end="\r")
    
        total_actor_loss += policy_loss.detach()
        total_critic_loss += critic_loss.detach()

    take_step()
    print(f"actor loss: {total_actor_loss:>7f}, critic loss: {total_critic_loss:>7f}")

    torch.save(agent.actor.state_dict(), "./parameters/actor")
    torch.save(agent.critic.state_dict(), "./parameters/critic")
    return total_actor_loss, total_critic_loss


total_episodes = 300
for episode in range(0,total_episodes, 3):

    for i in range(3): #Collect 3 episodes
        print("Episode: ", episode+i)

        env.reset()
        env.step()
        print("Hello")
        behavior_name = list(env.behavior_specs)[0] 
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        state = torch.tensor(decision_steps.obs[0], dtype=torch.float64).squeeze().permute(2,0,1)
        agent.memory = state.reshape(1, state.shape[0], 1, *state.shape[1:]).repeat(1, 1, retention_time, 1, 1)
        for t in range(retention_time):
            agent.replay_buffer.push(state, None, None, None, None)

        episode_reward = 0
        
        done = False
        for t_step in range(T_MAX):
            print(t_step)
            if done:
                print("Broke")
                break

            epsilon = max(0.1, 0.1*(10-episode))
            
            #if random.uniform(0, 1) < epsilon:
            action = np.random.rand(1,3) * 4 - 2
            #else:    
            #    action = agent.get_action()
            
            #action = get_input()
            env_action = ActionTuple(action)

            reward = 0
            for i in range(3):
                if not done:
                    env.set_actions(behavior_name, env_action)
                    env.step()
                    decision_steps, terminal_steps = env.get_steps(behavior_name)
                    done = len(decision_steps.reward)==0 or t_step == T_MAX
                    if not done:
                        reward += decision_steps.reward[0]
                        new_state = torch.Tensor(decision_steps.obs[0]).squeeze().permute(2,0,1)

                    else:
                        reward += terminal_steps.reward[0]
                        new_state = torch.Tensor(terminal_steps.obs[0]).squeeze().permute(2,0,1)

            agent.replay_buffer.push(state, torch.tensor(action), reward, new_state, done)
            
            agent.add_to_memory(state.cpu())            
            
            state = new_state
            episode_reward += reward


    dataset = my_dataset()
    train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    actor_loss, critic_loss = train_loop()
    actor_loss, critic_loss = actor_loss/len(dataset), critic_loss/len(dataset)

    torch.save(agent.actor.state_dict(), "./parameters/actor")
    torch.save(agent.critic.state_dict(), "./parameters/critic")
    
    rewards.append(episode_reward)
    if len(rewards) < 10:
        avg_rewards.append(np.mean(rewards))
    else:
        avg_rewards.append(np.mean(rewards[-10:]))

    writer.add_scalar('reward',
        rewards[-1],
        episode)
    writer.add_scalar('avg reward',
        avg_rewards[-1],
        episode)
    writer.add_scalar('critic loss',
        critic_loss,
        episode)
    writer.add_scalar('actor loss',
        actor_loss,
        episode)

    print('reward:', rewards[-1], "; avg reward:",avg_rewards[-1], '; critic loss:', critic_loss, "; actor loss:", actor_loss)



plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()