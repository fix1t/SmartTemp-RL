"""
    File: memory.py
    Author: Gabriel Biel

    Description: ReplayMemory class to store the experiences of the agent.
"""

import torch
import random
from collections import deque
import numpy as np

class ReplayMemory(object):
    def __init__(self, capacity):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, event):
        self.memory.append(event)

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            raise ValueError("Sample size greater than the number of elements in memory.")

        experiences = random.sample(self.memory, k=batch_size)

        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        return states, next_states, actions, rewards, dones
