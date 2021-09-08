#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from BrainClass import BrainClass

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[2]:


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
            weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps,
                weight_decay=weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


# In[3]:


class ActorCritic(nn.Module):
    def __init__(self, gamma=0.99):
        super(ActorCritic, self).__init__()

        self.gamma = gamma

        self.brain = BrainClass(n_heads=16)

        self.rewards = []
        self.actions = []
        self.states = []
        self.values = []

    def remember(self, state, action, reward, action_dist, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        
        self.action_dists.append(action_dist)
        self.values.append(value)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        
        self.action_dists = []
        self.values = []

    def forward(self, observation):
        action_dist, value = self.brain(observation, "PushBlock")
        return action_dist, value

    def calc_R(self, done):
        states = torch.tensor(self.states, dtype=torch.float)
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
        states = torch.tensor(self.states, dtype=torch.float)
        actions = torch.tensor(self.actions, dtype=torch.float)

        returns = self.calc_R(done)

        values = self.values.squeeze()
        critic_loss = (returns-values)**2

        e1 = - ((action_dist[[0,2]:] - actions)**2) / (2*action_dist[[1,3]:].clamp(min=1e-3))
        e2 = - torch.log(torch.sqrt(2 * math.pi * action_dist[[1,3]:]))
        actor_loss = e1 + e2

        total_loss = (critic_loss + actor_loss).mean()
    
        return total_loss

    def choose_action(self, observation):
        print("Hello11", observation.shape)
        action_dist, value = self.forward(observation)
        print("Hello22")
        forward_force = torch.normal(action_dist[0], action_dist[1])
        angular_velocity = torch.normal(action_dist[2], action_dist[3])
        
        return forward_force, angular_velocity, action_dist, value


# In[4]:


class Agent(mp.Process):
    def __init__(self, global_actor_critic, optimizers, 
                gamma, lr, name, global_ep_idx, env_path, worker_id):
        super(Agent, self).__init__()
        self.local_actor_critic = ActorCritic(gamma)
        self.global_actor_critic = global_actor_critic
        self.name = 'w%02i' % name
        self.episode_idx = global_ep_idx
        self.env = UnityEnvironment(worker_id=worker_id, file_name=env_path,  seed=1, side_channels=[])
        self.brain_optimizer = optimizers[0]
        self.task_memory_optimizer = optimizers[0]
        
        
    def run(self):
        t_step = 1
        while self.episode_idx.value < N_GAMES:
            done = False
            self.env.reset()
            self.env.step()
            behavior_names = self.env.behavior_specs.keys()
            behavior_name = list(self.env.behavior_specs)[0] 
            score = 0
            self.local_actor_critic.clear_memory()
            while not done:
                decision_steps, terminal_steps = self.env.get_steps(behavior_name)
                observation = torch.Tensor(decision_steps.obs[0]).squeeze().permute(2,0,1)
                print("111")
                forward_force, angular_velocity, action_dist, value = self.local_actor_critic.choose_action(observation)
                print("222")
                env_action = ActionTuple(np.array(action), dtype=np.float32)
                print("333")
                self.env.set_actions(behavior_name, action)
                print("444")
                self.env.step()
                print("555")
                score += reward
                self.local_actor_critic.remember(state, [forward_force, angular_velocity], reward, action_dist, value)
                
                if t_step % T_MAX == 0 or done:
                    loss = self.local_actor_critic.calc_loss(done)
                    
                    self.brain_optimizer.zero_grad()
                    self.task_memory_optimizer.zero_grad()
                    
                    loss.backward()
                    for local_param, global_param in zip(
                            self.local_actor_critic.parameters(),
                            self.global_actor_critic.parameters()):
                        global_param._grad = local_param.grad
                        
                    self.brain_optimizer.step()
                    self.task_memory_optimizer.step()
                    
                    self.local_actor_critic.load_state_dict(
                            self.global_actor_critic.state_dict())
                    self.local_actor_critic.clear_memory()
                t_step += 1
            with self.episode_idx.get_lock():
                self.episode_idx.value += 1
            print(self.name, 'episode ', self.episode_idx.value, 'reward %.1f' % score)


# In[ ]:


lr = 1e-4
task = "PushBlock"
N_GAMES = 3000
T_MAX = 5
global_actor_critic = ActorCritic()
global_actor_critic.share_memory()
brain_optim = SharedAdam(global_actor_critic.brain.get_brain_parameters(), lr=lr, 
                    betas=(0.92, 0.999))
task_memory_optim = SharedAdam(global_actor_critic.brain.get_task_memory_parameters(task), lr=lr, 
                    betas=(0.92, 0.999))
global_ep = mp.Value('i', 0)

workers = [Agent(global_actor_critic,
                [brain_optim, task_memory_optim],
                gamma=0.99,
                lr=lr,
                name=i,
                global_ep_idx=global_ep,
                env_path = "./Scenes/PushBlockScene",
                worker_id=i) for i in range(1)]
[w.start() for w in workers]
[w.join() for w in workers]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




