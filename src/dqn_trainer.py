import os
from collections import deque
import torch
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from algorithms.dql.agent import Agent as DQL
from algorithms.dql.network import Network
from algorithms.tools.logger import Logger
from env.environment import TempRegulationEnv


number_episodes = 1500
maximum_number_timesteps_per_episode = 4 * 24 * 7
epsilon_starting_value  = 1.0
epsilon_ending_value  = 0.01
epsilon_decay_value  = 0.995
epsilon = epsilon_starting_value
scores_on_100_episodes = deque(maxlen = 100)
print('Number of episodes: ', number_episodes)
print('Maximum number of timesteps per episode: ', maximum_number_timesteps_per_episode)
print('Epsilon starting value: ', epsilon_starting_value)
print('Epsilon ending value: ', epsilon_ending_value)
print('Epsilon decay value: ', epsilon_decay_value)
print('Epsilon: ', epsilon)

env = TempRegulationEnv(start_from_random_day=True)
state_shape = env.observation_space.shape
state_size = env.observation_space.shape[0]
number_actions = env.action_space.n
agent = DQL(env=env,policy_class=Network)
print('State shape: ', state_shape)
print('State size: ', state_size)
print('Number of actions: ', number_actions)


try:
    agent.train(number_episodes)
except KeyboardInterrupt:
    print("\nTraining interrupted. Saving current agent...")
finally:
    Logger().save_agent(agent)
    print("Agent saved successfully.")
    env.close()
    Logger().plot_scores()
    print("Scores plotted successfully. Results are saved in out/plots folder.")
