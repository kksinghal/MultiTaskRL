import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math 
from Agent import Agent

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class train_loop:
    def __init__(self, agent, env_path, task, gamma=0.99, lr=1e-4):

        self.gamma = gamma
        self.task = task

        self.rewards = []
        self.action_dists = []
        self.actions = []
        self.values = []

        self.agent = agent
        self.env = UnityEnvironment(file_name=env_path,  seed=1, side_channels=[])

        self.brain_optimizer = torch.optim.Adam(self.agent.get_brain_parameters(), lr=lr, 
                    betas=(0.92, 0.999))
        self.task_memory_optimizer = torch.optim.Adam(self.agent.get_task_memory_parameters(task), lr=lr, 
                            betas=(0.92, 0.999))

    def remember(self, reward, action_dist, action, value):
        self.rewards.append(reward)
        self.action_dists.append(action_dist)
        self.actions.append(action)
        self.values.append(value)


    def clear_memory(self):
        self.rewards = [] 
        self.action_dists = []
        self.actions = []
        self.values = []

    
    def calc_R(self, done):
        v = self.values

        R = v[-1]*(1-int(done))

        batch_return = []
        for reward in self.rewards[::-1]:
            R = reward + self.gamma*R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = torch.tensor(batch_return, dtype=torch.float)

        return batch_return


    def calc_loss(self, done):
        action_dists = torch.tensor(self.action_dists, dtype=torch.float)
        actions = torch.tensor(self.actions, dtype=torch.float)

        returns = self.calc_R(done)

        values = self.values.squeeze()
        critic_loss = (returns-values)**2

        e1 = - ((action_dists[[0,2]:] - actions)**2) / (2*action_dists[[1,3]:].clamp(min=1e-3))
        e2 = - torch.log(torch.sqrt(2 * math.pi * action_dists[[1,3]:]))
        actor_loss = e1 + e2

        total_loss = (critic_loss + actor_loss).mean()
    
        return total_loss


    def choose_action(self, observation, task):
        action_dist, value = self.agent(observation, task)
        forward_force = torch.normal(action_dist[0], action_dist[1])
        angular_velocity = torch.normal(action_dist[2], action_dist[3])
        
        return forward_force, angular_velocity, action_dist, value


    def run(self, N_GAMES, T_MAX):
        t_step = 1
        for episode_idx in range(N_GAMES):
            done = False
            self.env.reset()
            self.env.step()
            behavior_names = self.env.behavior_specs.keys()
            behavior_name = list(self.env.behavior_specs)[0] 
            score = 0

            while not done:
                decision_steps, terminal_steps = self.env.get_steps(behavior_name)
                score += decision_steps.reward[0]
                observation = torch.Tensor(decision_steps.obs[0]).squeeze().permute(2,0,1)
                forward_force, angular_velocity, action_dist, value = self.choose_action(observation, self.task)
                action = [forward_force, angular_velocity]
                env_action = ActionTuple(np.array([action], dtype=np.float32))
                self.env.set_actions(behavior_name, env_action)
                self.env.step()
                self.remember(decision_steps.reward[0], action_dist, action, value)
                
                if t_step % T_MAX == 0 or done:
                    loss = self.calc_loss(done)
                    
                    self.brain_optimizer.zero_grad()
                    self.task_memory_optimizer.zero_grad()
                    
                    loss.backward()

                    self.brain_optimizer.step()
                    self.task_memory_optimizer.step()
                    
                    self.clear_memory()

                t_step += 1

            print('episode ', episode_idx, 'reward %.1f' % score)

lr=1e-4
agent = Agent(n_heads=16)
env_path = None#"./Scenes/PushBlockScene/UnityEnvironment"
training_loop = train_loop(agent, env_path, task="PushBlock", lr=lr)
training_loop.run(N_GAMES=1000, T_MAX=5000)