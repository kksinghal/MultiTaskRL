import numpy as np
from collections import deque
import random
import torch


class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
            self.mu, self.sigma)


class ReplayBuffer:
    def __init__(self, max_size, retention_time):
        self.max_size = max_size
        self.retention_time = retention_time
        self.buffer = deque(maxlen=max_size)
        self.sampling_ids = np.empty((0,), dtype=np.int32)
    
    def push(self, state, action, reward, next_state, done):
        if next_state is not None:
            if len(self.buffer) == self.max_size:
                self.sampling_ids = np.append(self.sampling_ids, self.max_size)
            else:
                self.sampling_ids = np.append(self.sampling_ids, len(self.buffer))

        if len(self.buffer) == self.max_size:
            self.sampling_ids -= 1 
            self.sampling_ids = self.sampling_ids[self.sampling_ids>=self.retention_time]
            
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size, retention_time):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch_ids = np.random.choice(self.sampling_ids, batch_size)

        for id in batch_ids:
  
            state, action, reward, next_state, done = self.buffer[id]


            current_state_memory = state.reshape(state.shape[0], 1, *state.shape[1:])
            for _id in range(id-1, id-retention_time, -1):
                old_state = self.buffer[_id][0]
                old_state = old_state.reshape(old_state.shape[0], 1, *old_state.shape[1:])

                current_state_memory = torch.cat((current_state_memory, old_state), dim=1)

            state_batch.append(current_state_memory)
            
            next_state = next_state.reshape(next_state.shape[0], 1, *next_state.shape[1:])
            next_state_memory = torch.cat((next_state, current_state_memory[:,:-1]), dim=1)
            next_state_batch.append(next_state_memory)

            action_batch.append(action[0])
            reward_batch.append(reward)
            done_batch.append(done)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)