from envs.pz_simpy import env
from gymnasium.spaces import Discrete

env = env(render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        a_space = env.action_space(agent)
        #a_space = Discrete(3)
        action = a_space.sample()

    env.step(action)
env.close()