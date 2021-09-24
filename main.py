
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Agent import Agent
from utils import *

import torch

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

env_path = None#"./Scenes/GoalieVS2Striker/UnityEnvironment"
env = UnityEnvironment(file_name=env_path,  seed=1, side_channels=[])

retention_time = 2
agent = Agent(retention_time)
batch_size = 16
rewards = []
avg_rewards = []

for episode in range(50):
    env.reset()
    env.step()

    behavior_name = list(env.behavior_specs)[0] 
    decision_steps, terminal_steps = env.get_steps(behavior_name)

    state = torch.Tensor(decision_steps.obs[0]).squeeze().permute(2,0,1)
    agent.memory = state.reshape(1, *state.shape).repeat(retention_time, *state.shape)
    print(agent.memory.shape)
    episode_reward = 0
    
    for step in range(300):
        action = agent.get_action()
        if episode < 50:
            action = torch.rand((3,)) * 50
        else:
            action = torch.normal(mean=action, std=action*0.2) 

        env_action = ActionTuple(action.detach().numpy())


        reward = 0
        for i in range(3):
            if not done:
                env.set_actions(behavior_name, env_action)
                env.step()
                decision_steps, terminal_steps = env.get_steps(behavior_name)
                done = len(decision_steps.reward)==0 or t_step == T_MAX
                if not done:
                    reward += decision_steps.reward[0]
                else:
                    reward += terminal_steps.reward[0]

        new_state = torch.Tensor(decision_steps.obs[0]).squeeze().permute(2,0,1)

        agent.memory.push(state, action, reward, new_state, done)
        
        if len(agent.memory) > batch_size:
            agent.update(batch_size)        
        
        state = new_state
        episode_reward += reward

        if done:
            sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
            break

    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-10:]))

plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()