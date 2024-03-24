"""Uses Stable-Baselines3 to train agents in the SimPy-RL environment where the agents goal is
optimal rerouting of traffic under DOS attack. This is training all the routers with different 
action spaces. There is performance bottleneck due to the padding of uneven action space.

For more information, see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

and the SimPy RL environment see https://github.com/NREL/DSS-SimPy-RL

Author: Abhijeet Sahu (https://github.com/Abhijeet1990)
"""
from __future__ import annotations

import glob
import os
import random
import time
from statistics import mean 
from SuperSuit import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy, MlpPolicy

from envs.pz_simpy_complex import env
from pettingzoo.utils.conversions import aec_to_parallel

def train(env, steps: int =1000):
    env.reset()
    env = aec_to_parallel(env)
    #env = ss.black_death_v3(env)
    env = ss.multiagent_wrappers.pad_action_space_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 7, num_cpus=1, base_class="stable_baselines3")
    model = PPO(MlpPolicy,
                env=env,
                batch_size=64,
                ent_coef=0.0,
                learning_rate=0.0003,
                n_epochs=10,
                n_steps=64,
                verbose=3)
    model.learn(total_timesteps=steps,progress_bar=True)

    #print("Model has been saved.")

    print("Finished training.")

    env.close()
    return model
    
def eval(env,model,num_games: int = 2,random:bool = False):

    rewards = {agent: 0 for agent in env.possible_agents}

    # Note: we evaluate here using an AEC environments, to allow for easy A/B testing against random policies
    # For example, we can see here that using a random agent for archer_0 results in less points than the trained agent
    game_lengths = []
    for i in range(num_games):
        #print(f'Episode {i+1}')
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)
        episode_len = 0
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for a in env.agents:
                rewards[a] += env.rewards[a]

            if termination or truncation:
                break
            else:
                if random:
                    act = env.action_space(agent).sample()
                else:
                    #act = model.predict(obs, deterministic=True)[0]
                    act = model.predict(obs)[0]
            #print(f'Agent:{agent} action:{act}')
            if agent in ['R2','R4','R5']:
                if act > 1:
                    act = 1
            #Because there is only one forwarding
            if agent in ['R1','R6','R7']:
                act = 0
            episode_len+=1
            env.step(act)
        per_agent_epi_len = episode_len/len(env.possible_agents)
        # succesful episode
        if per_agent_epi_len < 20:
            game_lengths.append(per_agent_epi_len)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    avg_reward_per_agent = {
        agent: rewards[agent] / num_games for agent in env.possible_agents
    }
    print(f"Avg reward: {avg_reward}")
    print("Avg reward per agent, per game: ", avg_reward_per_agent)
    print("Full rewards: ", rewards)
    if len(game_lengths) > 0:
        print(f"Episode length stats of succesful episode: {mean(game_lengths)} max : {max(game_lengths)} min : {min(game_lengths)} ", )
    else:
        print("No succesful episodes")
    print("success rate ",len(game_lengths)/num_games)
    return avg_reward



if __name__ == "__main__":
    env = env(render_mode="human")
    model = train(env,steps=100000)
    eval(env,model,num_games=100,random=True)
    eval(env,model,num_games=100,random=False)
