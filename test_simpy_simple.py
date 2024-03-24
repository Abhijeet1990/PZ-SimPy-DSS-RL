"""Uses Stable-Baselines3 to train agents in the SimPy-RL environment where the agents goal is
optimal rerouting of traffic under DOS attack for a small network. This is training all the routers with different 
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

from SuperSuit import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy, MlpPolicy

from envs.pz_simpy_simple import env
from pettingzoo.utils.conversions import aec_to_parallel

def train(env, steps: int =10000):
    env.reset()
    env = aec_to_parallel(env)
    #env = ss.black_death_v3(env)
    env = ss.multiagent_wrappers.pad_action_space_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 5, num_cpus=1, base_class="stable_baselines3")
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
    
def eval(env,model,num_games: int = 2):

    rewards = {agent: 0 for agent in env.possible_agents}

    # Note: we evaluate here using an AEC environments, to allow for easy A/B testing against random policies
    # For example, we can see here that using a random agent for archer_0 results in less points than the trained agent
    for i in range(num_games):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for a in env.agents:
                rewards[a] += env.rewards[a]

            if termination or truncation:
                break
            else:
                if agent == env.possible_agents[0]:
                    act = env.action_space(agent).sample()
                else:
                    act = model.predict(obs, deterministic=True)[0]
            if agent == 'R2':
                if act > 1:
                    act = random.randint(0,1)
            # Because there is only one forwarding
            if agent in ['R3','R4','R5']:
                act = 0
            
            env.step(act)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    avg_reward_per_agent = {
        agent: rewards[agent] / num_games for agent in env.possible_agents
    }
    print(f"Avg reward: {avg_reward}")
    print("Avg reward per agent, per game: ", avg_reward_per_agent)
    print("Full rewards: ", rewards)
    return avg_reward

if __name__ == "__main__":
    env = env(render_mode="human")
    model = train(env)
    eval(env,model,num_games=2)
