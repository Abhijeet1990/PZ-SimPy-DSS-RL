"""Uses Agile RL framework to train agents in the SimPy-RL environment where the agents goal is
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
import torch
import numpy as np
from statistics import mean 
from SuperSuit import supersuit as ss
from envs.pz_simpy_complex import env
from pettingzoo.utils.conversions import aec_to_parallel
from agilerl.algorithms.matd3 import MATD3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cenv = env(render_mode="human")
cenv = aec_to_parallel(cenv)
cenv.reset()

# Configure the multi-agent algo input arguments
try:
    state_dim = [cenv.observation_space(agent).n for agent in cenv.agents]
    one_hot = True
except Exception:
    state_dim = [cenv.observation_space(agent).shape for agent in cenv.agents]
    one_hot = False
try:
    action_dim = [cenv.action_space(agent).n for agent in cenv.agents]
    discrete_actions = True
    max_action = None
    min_action = None
except Exception:
    action_dim = [cenv.action_space(agent).shape[0] for agent in cenv.agents]
    discrete_actions = False
    max_action = [cenv.action_space(agent).high for agent in cenv.agents]
    min_action = [cenv.action_space(agent).low for agent in cenv.agents]

 # Append number of agents and agent IDs to the initial hyperparameter dictionary
n_agents = cenv.num_agents
agent_ids = cenv.agents

# Instantiate an MATD3 object
matd3 = MATD3(
        state_dim,
        action_dim,
        one_hot,
        n_agents,
        agent_ids,
        max_action,
        min_action,
        discrete_actions,
        device=device,
    )

# Load the saved algorithm into the MADDPG object
path = "./models/MATD3/simpy_MATD3_trained_agent_v2.pt"
matd3.loadCheckpoint(path)

# Define test loop parameters
episodes = 100  # Number of episodes to test agent on
max_steps = 25  # Max number of steps to take in the environment in each episode

rewards = []  # List to collect total episodic reward
frames = []  # List to collect frames
indi_agent_rewards = {
    agent_id: [] for agent_id in agent_ids
}  # Dictionary to collect inidivdual agent rewards

# Test loop for inference
ep_lengths = []
for ep in range(episodes):
    state, info = cenv.reset()
    agent_reward = {agent_id: 0 for agent_id in agent_ids}
    score = 0
    ep_length = 0
    for _ in range(max_steps):
        agent_mask = info["agent_mask"] if "agent_mask" in info.keys() else None
        env_defined_actions = (
            info["env_defined_actions"]
            if "env_defined_actions" in info.keys()
            else None
        )

        # Get next action from agent
        cont_actions, discrete_action = matd3.getAction(
            state,
            epsilon=1,
            agent_mask=agent_mask,
            env_defined_actions=env_defined_actions,
        )
        if matd3.discrete_actions:
            action = discrete_action
        else:
            action = cont_actions

        # Take action in environment
        state, reward, termination, truncation, info = cenv.step(action)
        ep_length+=1

        # Save agent's reward for this step in this episode
        for agent_id, r in reward.items():
            agent_reward[agent_id] += r

        # Determine total score for the episode and then append to rewards list
        score = sum(agent_reward.values())

        # Stop episode if any agents have terminated
        if any(truncation.values()) or any(termination.values()):
            break

    rewards.append(score)

    # Record agent specific episodic reward for each agent
    for agent_id in agent_ids:
        indi_agent_rewards[agent_id].append(agent_reward[agent_id])

    #print("-" * 15, f"Episode: {ep}", "-" * 15)
    #print("Episodic Reward: ", rewards[-1])
    #print("Episode Length: ",ep_length)
    ep_lengths.append(ep_length)
    # for agent_id, reward_list in indi_agent_rewards.items():
    #     print(f"{agent_id} reward: {reward_list[-1]}")

print('Average Episode Lengths: ',mean(ep_lengths))
cenv.close()


