import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from smart_home_env import SmartHomeTempControlEnv
import tqdm

# Define hyperparameters for the A3C agent
learning_rate = 1e-4  # Learning rate for optimizer
discount_factor = 0.99  # Discount factor for future rewards
number_environments = 10  # Number of parallel environments (not used in this code snippet)

class A3C():
    """
    A class that represents the A3C agent. It includes methods for selecting actions,
    and updating the policy and value networks based on received rewards.
    """
    def __init__(self, action_size):
        """
        Initializes the A3C agent.

        Parameters:
            action_size (int): The size of the action space.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Set device for training
        self.action_size = action_size  # Number of actions
        self.network = Network(action_size).to(self.device)  # Initialize the policy and value network
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)  # Define optimizer

    def act(self, state):
        """
        Chooses an action based on the current state using the policy network.

        Parameters:
            state (np.array): The current state of the environment.

        Returns:
            np.array: The chosen action.
        """
        # Convert state to tensor and get action probabilities
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_values, _ = self.network(state)
        policy = F.softmax(action_values, dim=-1)

        # Sample action from the action distribution
        return np.array([np.random.choice(len(p), p=p) for p in policy.detach().cpu().numpy()])

    def step(self, state, action, reward, next_state, done):
        """
        Performs a single step of training on the agent.

        Parameters:
            state (np.array): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (np.array): The next state.
            done (bool): Whether the episode has ended.
        """
        # Convert parameters to tensors
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)

        # Calculate loss and perform backpropagation
        action_values, state_value = self.network(state)
        _, next_state_value = self.network(next_state)

        # Compute target and advantage
        target_state_value = reward + discount_factor * next_state_value * (1 - done)
        advantage = target_state_value - state_value

        # Calculate policy and value loss
        probs = F.softmax(action_values, dim=-1)
        logprobs = F.log_softmax(action_values, dim=-1)
        entropy = -(probs * logprobs).sum(-1).mean()  # Entropy regularization
        action_log_probs = logprobs.gather(1, action.unsqueeze(-1)).squeeze(-1)
        actor_loss = -(action_log_probs * advantage.detach()).mean() - 0.001 * entropy
        critic_loss = F.mse_loss(state_value, target_state_value.detach())

        # Update network weights
        total_loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

class Network(nn.Module):
    """
    Neural network class for A3C agent adapted for vector input with additional hidden layers.
    """
    def __init__(self, input_size, action_size):
        """
        Initializes the neural network for vector inputs with more complexity.

        Parameters:
            input_size (int): The size of the input vector.
            action_size (int): The size of the action space.
        """
        super(Network, self).__init__()
        # Definitio of the fully connected layers
        self.fc1 = nn.Linear(input_size, 256)  # First hidden layer
        self.fc2 = nn.Linear(256, 128)  # Second hidden layer
        self.fc3 = nn.Linear(128, 64)   # Third hidden layer, for more complexity

        self.fc_action = nn.Linear(64, action_size)  # Action output layer
        self.fc_value = nn.Linear(64, 1)  # State value output layer

    def forward(self, state):
        """
        Forward pass through the network for vector inputs with multiple hidden layers.

        Parameters:
            state (torch.Tensor): The input state vector.

        Returns:
            torch.Tensor: The action values (logits).
            torch.Tensor: The state value.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action_values = self.fc_action(x)
        state_value = self.fc_value(x).squeeze(-1)  # Ensure state value has correct shape
        return action_values, state_value

def evaluate(agent, env, n_episodes = 1):
  episodes_rewards = []
  for _ in range(n_episodes):
    state, _ = env.reset()
    total_reward = 0
    while True:
      action = agent.act(state)
      state, reward, done, info, _ = env.step(action[0])
      total_reward += reward
      if done:
        break
    episodes_rewards.append(total_reward)
  return episodes_rewards

def make_env():
  return SmartHomeTempControlEnv(start_from_random_day=True)

class EnvBatch:

  def __init__(self, n_envs = number_environments):
    self.envs = [make_env() for _ in range(n_envs)]

  def reset(self):
    _states = []
    for env in self.envs:
      _states.append(env.reset()[0])
    return np.array(_states)

  def step(self, actions):
    next_states, rewards, dones, infos, _ = map(np.array, zip(*[env.step(a) for env, a in zip(self.envs, actions)]))
    for i in range(len(self.envs)):
      if dones[i]:
        next_states[i] = self.envs[i].reset()[0]
    return next_states, rewards, dones, infos

env = make_env()
env_batch = EnvBatch(number_environments)
batch_states = env_batch.reset()
agent = A3C(env_batch.envs[0].action_space.n)

with tqdm.trange(0, 3001) as progress_bar:
  for i in progress_bar:
    batch_actions = agent.act(batch_states)
    batch_next_states, batch_rewards, batch_dones, _ = env_batch.step(batch_actions)
    batch_rewards *= 0.01
    agent.step(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones)
    batch_states = batch_next_states
    if i % 1000 == 0:
      print("Average agent reward: ", np.mean(evaluate(agent, env, n_episodes = 10)))
