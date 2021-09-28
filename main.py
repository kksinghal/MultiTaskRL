
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

retention_time = 3
agent = Agent(retention_time)
batch_size = 4
rewards = []
avg_rewards = []

T_MAX = 300
for episode in range(50):
    env.reset()
    env.step()

    behavior_name = list(env.behavior_specs)[0] 
    decision_steps, terminal_steps = env.get_steps(behavior_name)

    state = torch.Tensor(decision_steps.obs[0]).squeeze().permute(2,0,1)
    agent.memory = state.reshape(1, state.shape[0], 1, *state.shape[1:]).repeat(1, 1, retention_time, 1, 1)
    for t in range(retention_time):
        agent.replay_buffer.push(state, None, None, None, None)
 
    episode_reward = 0
    
    done = False
    for t_step in range(T_MAX):
        print(t_step)
        action = agent.get_action()
        if episode < 50:
            action = torch.rand((1,3))
        else:
            action = torch.normal(mean=action, std=action*0.2) 

        env_action = ActionTuple(action.detach().cpu().numpy())


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

        agent.replay_buffer.push(state, action, reward, new_state, done)
        
        agent.add_to_memory(state)
        
        if len(agent.replay_buffer) > batch_size + retention_time:
            print("yes0")
            agent.update(batch_size)        
            print("yes1")
        
        state = new_state
        episode_reward += reward

        if done:
            sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
            break

    if episode%5 == 0:
        torch.save(agent.actor.state_dict(), "./parameters/actor")
        torch.save(agent.critic.state_dict(), "./parameters/critic")

    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-10:]))

plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()