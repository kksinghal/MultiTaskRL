import os
import sys
import numpy as np
import pandas as pd
from Agent import Agent
from utils import *
import random
import keyboard
import torch

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple


env_path = None#"./Scenes/PushBlock_Win_slow/UnityEnvironment"
env = UnityEnvironment(file_name=env_path,  seed=1, side_channels=[])


retention_time = 3
agent = Agent(retention_time)

T_MAX = 300


def get_input():
    forward_force_mean = 0 
    right_force_mean = 0 
    angular_velocity_mean = 0
    try:
        if keyboard.is_pressed('w'):
            forward_force_mean = 0.8
        if keyboard.is_pressed('f'):
            forward_force_mean = -0.8
        if keyboard.is_pressed('s'):
            forward_force_mean = -0.8
        
        if keyboard.is_pressed('d'):
            right_force_mean = 0.8
        if keyboard.is_pressed('a'):
            right_force_mean = -0.8

        if keyboard.is_pressed('m'):
            angular_velocity_mean = 0.8
        if keyboard.is_pressed('n'):
            angular_velocity_mean = -0.8
    except:
        print("except")
    return np.array([[forward_force_mean, right_force_mean, angular_velocity_mean]])*2

buffer_data = pd.read_csv("./data/buffer_data.csv")

total_episodes = 15
for episode in range(total_episodes):
    
    print("Episode: ", episode)

    env.reset()
    env.step()

    behavior_name = list(env.behavior_specs)[0] 
    decision_steps, terminal_steps = env.get_steps(behavior_name)

    state = torch.tensor(decision_steps.obs[0], dtype=torch.float64).squeeze().permute(2,0,1)
    for t in range(retention_time):
        agent.replay_buffer.push(state, None, None, None, None)

    done = False
    for t_step in range(T_MAX):
        if done:
            break

        action = get_input()
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
        state = new_state

        id = len(buffer_data)
        torch.save(state, "./data/states/"+str(id)+".pt")
        torch.save(new_state, "./data/next_states/"+str(id)+".pt")

        data = pd.DataFrame([[id, *(action[0]), reward, done]], columns=buffer_data.columns)
        buffer_data = buffer_data.append(data)
        buffer_data.to_csv("./data/buffer_data.csv", index=False)

        np.save("./data/sampling_ids", agent.replay_buffer.sampling_ids)
