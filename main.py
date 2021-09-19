import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math 
from Agent import Agent

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('tensorboard/loss')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class train_loop:
    def __init__(self, agent, env_path, task, gamma=0.99, lr=1e-4):

        self.gamma = gamma
        self.task = task

        self.rewards = []
        self.action_dists = []
        self.actions = []
        self.values = []

        self.agent = agent.to(device)

        self.env = UnityEnvironment(file_name=env_path,  seed=1, side_channels=[])

        self.brain_optimizer = torch.optim.Adam(self.agent.get_brain_parameters(), lr=lr*1e-2, 
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
        action_dists = torch.stack(self.action_dists)

        actions = torch.stack([torch.stack(i) for i in self.actions])

        returns = self.calc_R(done)

        values = torch.Tensor(self.values)
        critic_loss = ((returns-values)**2).sum()
        #loss_value_v = F.mse_loss(returns, self.values)        

        

        adv_v = returns - values 

        e1 = - torch.sum(((action_dists[:,[0,2,4]] - actions)**2) / (2*action_dists[:,[1,3,5]]))
        e2 = - torch.sum(torch.log(torch.sqrt(2 * math.pi * action_dists[:,[1,3,5]])))
        log_prob = e1 + e2

        log_prob_v = adv_v * log_prob

        loss_policy_v = -log_prob_v.mean()
        ENTROPY_BETA = 1
        entropy_loss_v = ENTROPY_BETA * (-(torch.log(2*math.pi*action_dists[:,[1,3,5]])+1)/2).mean()

        total_loss = (critic_loss + entropy_loss_v + loss_policy_v).mean()
        return total_loss


    def choose_action(self, observation, task):
        action_dist, value = self.agent(observation, task)
        forward_force = torch.normal(action_dist[0], action_dist[1]) * 1000
        right_force = torch.normal(action_dist[2], action_dist[3]) * 1000
        angular_velocity = torch.normal(action_dist[4], action_dist[5]) * 100
        
        return forward_force, right_force, angular_velocity, action_dist, value


    def run(self, N_GAMES, T_MAX):

        for episode_idx in range(N_GAMES):
            done = False
            self.env.reset()
            self.env.step()
            behavior_names = self.env.behavior_specs.keys()
            behavior_name = list(self.env.behavior_specs)[0] 
            score = 0

            t_step = 0
            while not done:

                decision_steps, terminal_steps = self.env.get_steps(behavior_name)

                observation = torch.Tensor(decision_steps.obs[0]).squeeze().permute(2,0,1)
                observation = observation.to(device)
                out = self.choose_action(observation, self.task)
                
                forward_force, right_force, angular_velocity, action_dist, value = [i.cpu() for i in out]

                del observation, out
                
                action = [forward_force, right_force, angular_velocity]
                env_action = ActionTuple(torch.tensor([action]).detach().numpy())
                
                tmp_score = 0
                for i in range(3):
                    if not done:
                        self.env.set_actions(behavior_name, env_action)
                        self.env.step()
                        decision_steps, terminal_steps = self.env.get_steps(behavior_name)
                        done = len(decision_steps.reward)==0 or t_step == T_MAX
                        if not done:
                            tmp_score += decision_steps.reward[0]
                        else:
                            tmp_score += terminal_steps.reward[0]
    
                self.remember(tmp_score, action_dist, action, value)

                score += tmp_score 

                if done:
                    if t_step == T_MAX:
                        score -= 1
                    else:
                        score += terminal_steps.reward[0]

                    loss = self.calc_loss(done)

                    writer.add_scalar('training loss',
                            loss / 1000,
                            episode_idx)
                    
                    self.brain_optimizer.zero_grad()
                    self.task_memory_optimizer.zero_grad()
                    
                    loss.backward()

                    self.brain_optimizer.step()
                    self.task_memory_optimizer.step()
                    
                    self.clear_memory()
                    
                t_step += 1

            print('episode ', episode_idx, 'reward %.1f' % score)
            if (episode_idx+1) %10 == 0:
                self.agent.save_parameters()

lr=1e-4
agent = Agent(n_heads=16)
env_path = "./Scenes/GoalieVS2Striker/UnityEnvironment"
training_loop = train_loop(agent, env_path, task="goalieVSstrikers", lr=lr)
training_loop.run(N_GAMES=5000, T_MAX=330)