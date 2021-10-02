import torch
from torchvision.utils import save_image

import pandas as pd
import numpy as np

import os
import keyboard
from pathlib import Path


from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple


def calc_R(rewards, discount=0.95):

    R = 0
    batch_return = []
    for reward in rewards[::-1]:
        R = reward + discount*R
        batch_return.append(R)
    batch_return.reverse()

    return batch_return

    

N_GAMES = 5
T_MAX=330
env = UnityEnvironment(file_name=None,  seed=1, side_channels=[])

task = "PushBlock"
Path("./data/"+task).mkdir(parents=True, exist_ok=True)
already_collected_episodes = len(next(os.walk('./data/'+task))[1])

for episode_idx in range(already_collected_episodes, already_collected_episodes+N_GAMES):

    forward_force_means= [] 
    right_force_means= [] 
    angular_velocity_means = []
    rewards = []

    os.mkdir("./data/"+task+"/"+str(episode_idx))
    done = False
    env.reset()
    env.step()
    behavior_names = env.behavior_specs.keys()
    behavior_name = list(env.behavior_specs)[0] 
    score = 0

    t_step = 0
    output = []
    while not done:

        decision_steps, terminal_steps = env.get_steps(behavior_name)

        observation = torch.Tensor(decision_steps.obs[0]).squeeze().permute(2,0,1)

        save_image(observation, './data/'+task+"/"+str(episode_idx)+'/'+ str(t_step) +'.png')

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
        
        forward_force_means.append(forward_force_mean)
        right_force_means.append(right_force_mean)
        angular_velocity_means.append(angular_velocity_mean)

        action = [forward_force_mean, right_force_mean, angular_velocity_mean]
        env_action = ActionTuple(np.array([action]))

        tmp_score = 0
        for i in range(3):
            if not done:
                env.set_actions(behavior_name, env_action)
                env.step()
                decision_steps, terminal_steps = env.get_steps(behavior_name)
                done = len(terminal_steps.reward)!=0 or t_step == T_MAX
                if not done:
                    tmp_score += decision_steps.reward[0]
                else:
                    tmp_score += terminal_steps.reward[0]

        score += tmp_score
        rewards.append(tmp_score)
        

        if done:
            if t_step == T_MAX:
                score -= 1
            else:
                score += terminal_steps.reward[0]
            
            returns = calc_R(rewards)

            df = pd.DataFrame({
                "id": range(t_step+1),
                "forward_force_mean": forward_force_means, 
                "right_force_mean": right_force_means, 
                "angular_velocity_mean": angular_velocity_means, 
                "value": returns
            })

            df.to_csv("./data/"+task+"/"+str(episode_idx)+"/"+"output.csv", index=False)

            df2 = pd.DataFrame({
                "task": [task],
                "episode_idx": [episode_idx]
            })
            data = pd.read_csv("./data/metadata.csv", index_col=None)
            data = data.append(df2)
            data.to_csv("./data/metadata.csv", index=False)

            

        t_step += 1

    print('episode ', episode_idx, 'reward %.1f' % score)