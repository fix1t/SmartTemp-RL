import gym
import numpy as np
import sys
import random
from collections import deque
import torch
from agent.agent import Agent
from smart_home_env import SmartHomeTempControlEnv

env = SmartHomeTempControlEnv()
state_size = env.observation_space.shape[0]
number_actions = env.action_space.n
agent = Agent(state_size, number_actions)
print('State size: ', state_size)
print('Number of actions: ', number_actions)

if len(sys.argv) != 2:
    print("Usage: python agent_replay.py <filename>")
    sys.exit(1)

filename = sys.argv[1]
checkpoint = torch.load(filename)

agent.local_qnetwork.load_state_dict(checkpoint)

state = env.reset()
env.render()
total_reward = 0
done = False
while not done:
    action = agent.act(state, epsilon=0)  # Using epsilon=0 for evaluation
    state, reward, done, _ = env.step(action)
    total_reward += reward
    print("Action:", action, "Reward:", reward)
print("Total reward:", total_reward)
