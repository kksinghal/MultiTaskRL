import numpy as np
from collections import deque
import random
import torch


class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size, retention_time):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch_ids = random.sample(range(retention_time, len(self.buffer)), batch_size)

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