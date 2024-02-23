import gym
import os
import numpy as np
import random
from collections import deque
import torch
from agent.agent import Agent
from smart_home_env import SmartHomeTempControlEnv
from datetime import datetime

number_episodes = 3000
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

env = SmartHomeTempControlEnv()
state_shape = env.observation_space.shape
state_size = env.observation_space.shape[0]
number_actions = env.action_space.n
agent = Agent(state_size, number_actions)
print('State shape: ', state_shape)
print('State size: ', state_size)
print('Number of actions: ', number_actions)


def save_agent(agent, score, number_of_episodes):
    if not os.path.exists('agents'):
        os.makedirs('agents')
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f'agents/{current_time}_{int(score)}_in_{number_of_episodes}_eps.pth'
    torch.save(agent.local_qnetwork.state_dict(), filename)



for episode in range(1, number_episodes + 1):
  state = env.reset()
  score = 0
  for t in range(maximum_number_timesteps_per_episode):
    action = agent.act(state, epsilon)
    next_state, reward, done, _ = env.step(action)
    agent.step(state, action, reward, next_state, done)
    state = next_state
    score += reward
    if done:
      break
  scores_on_100_episodes.append(score)
  epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)
  print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)), end = "")
  if episode % 100 == 0:
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)))
  if np.mean(scores_on_100_episodes) >= 100.0:
    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)))
    break
save_agent(agent, np.mean(scores_on_100_episodes), episode)
