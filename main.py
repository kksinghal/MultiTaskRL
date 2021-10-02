import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Agent import Agent
from utils import *
import random
import keyboard
import torch

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('tensorboard/')

env_path = "./Scenes/PushBlock_Win/UnityEnvironment"
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

T_MAX = 300


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

total_episodes = 10
for episode in range(0,total_episodes, 3):

    for i in range(3): #Collect 3 episodes
        print("Episode: ", episode+i)

        env.reset()
        env.step()

        behavior_name = list(env.behavior_specs)[0] 
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        state = torch.tensor(decision_steps.obs[0], dtype=torch.float64).squeeze().permute(2,0,1)
        agent.memory = state.reshape(1, state.shape[0], 1, *state.shape[1:]).repeat(1, 1, retention_time, 1, 1)
        for t in range(retention_time):
            agent.replay_buffer.push(state, None, None, None, None)

        episode_reward = 0
        
        done = False
        for t_step in range(T_MAX):
            if done:
                break
            
            """
            if random.uniform(0, 1) < epsilon:
                action = np.random.rand(1,3) * 2 - 2
            else:    
                action = agent.get_action()
            """
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
            
            agent.add_to_memory(state.cpu())            
            
            state = new_state
            episode_reward += reward

    actor_loss = 0
    critic_loss = 0
    for i in range(250):
        actor_loss_, critic_loss_ = agent.update(batch_size)  
        actor_loss += actor_loss_     
        critic_loss += critic_loss_ 
        print("Epoch: " + str(i) + "/250, actor loss: "+str(actor_loss_) + ", critic loss: "+str(critic_loss_) , end="\r")    


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

    print('reward:', rewards[-1], "; avg reward:",avg_rewards[-1], '; critic loss:', critic_loss/t_step, "; actor loss:", actor_loss/t_step)



plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()