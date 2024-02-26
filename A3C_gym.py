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
number_environments = 12  # Number of parallel environments (not used in this code snippet)

class A3C():
    """
    A class that represents the A3C agent. It includes methods for selecting actions,
    and updating the policy and value networks based on received rewards.
    """
    def __init__(self, state_size ,action_size):
        """
        Initializes the A3C agent.

        Parameters:
            state_size (int): The size of the state space.
            action_size (int): The size of the action space.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Set device for training
        self.action_size = action_size  # Number of actions
        self.network = Network(state_size ,action_size).to(self.device)  # Initialize the policy and value network
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)  # Define optimizer

    def act(self, states):
        """
        Chooses actions for a batch of states using the policy network.

        Parameters:
            states (np.array): The current states of the environments.

        Returns:
            np.array: The chosen actions for each state.
        """
        # states = torch.tensor(states, dtype=torch.float32, device=self.device)
        # action_values, _ = self.network(states)

        # print("Action values: ")
        # print(action_values)
        # print(action_values.shape)

        # policy = F.softmax(action_values, dim=-1).cpu().detach().numpy()

        # # Sample actions for each state in the batch
        # print("Policy: ")
        # print(policy)
        # print(policy.shape)
        # actions = [np.random.choice(self.action_size, p=policy[i]) for i in range(policy.shape[0])]
        # return np.array(actions)
        if states.ndim == 1:  # Handling a single stWate passed to act
            states = np.atleast_2d(states)  # Convert to 2D array
        print(f"States:{states.shape} {states}")
        states = np.atleast_2d(states)
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        action_values, _ = self.network(states)
        print(f"Action values:{action_values.shape} {action_values}")
        policy = F.softmax(action_values, dim=-1).cpu().detach().numpy()
        print(f"Policy:{policy.shape} {policy}")
        # Sample actions for each state in the batch
        if policy.ndim == 1:  # Handling a single state passed to act
            actions = np.random.choice(self.action_size, p=policy)
        else:  # Handling a batch of states
            actions = [np.random.choice(self.action_size, p=policy[i]) for i in range(policy.shape[0])]
        print(f"Actions:{actions}")
        return actions if policy.ndim > 1 else [actions]  # Ensure output is always a list of actions

    def step(self, state, action, reward, next_state, done):
        """
        Performs a single step of training on the agent, updating both the policy (actor)
        and the value (critic) networks based on the received batch of experiences.

        Parameters:
            state (np.array): The current states for a batch of environments.
            action (np.array): The actions taken for each environment.
            reward (np.array): The rewards received for each environment.
            next_state (np.array): The next states for each environment.
            done (np.array): The done flags indicating whether each environment has finished.
        """
        batch_size = state.shape[0]  # Determine the batch size from the state shape
        # Convert all inputs to PyTorch tensors and move to the specified device
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.bool, device=self.device).to(dtype=torch.float32)

        # Forward pass through the network for both current and next states
        action_values, state_value = self.network(state)
        _, next_state_value = self.network(next_state)

        # Calculate the target value for each state using the Bellman equation
        target_state_value = reward + discount_factor * next_state_value * (1 - done)

        # Calculate the advantage as the difference between target and predicted state values
        advantage = target_state_value - state_value

        # Compute the log probabilities of the actions
        probs = F.softmax(action_values, dim=-1)
        logprobs = F.log_softmax(action_values, dim=-1)

        # Calculate the entropy to encourage exploration
        entropy = -torch.sum(probs * logprobs, dim=-1)

        # Select the log probabilities of the chosen actions
        action = torch.tensor(action, dtype=torch.long, device=self.device)  # Ensure action tensor is correct
        logp_actions = logprobs[np.arange(batch_size), action]

        # Calculate the losses for both actor and critic components
        actor_loss = -(logp_actions * advantage.detach()).mean() - 0.001 * entropy.mean()
        critic_loss = F.mse_loss(state_value, target_state_value.detach())

        # Combine the losses, perform backpropagation, and update the network weights
        total_loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

class Network(nn.Module):
    """
    Neural network class for A3C agent adapted for vector input with additional hidden layers.
    """
    def __init__(self, state_size, action_size):
        """
        Initializes the neural network for vector inputs with more complexity.

        Parameters:
            state_size (int): The size of the input vector.
            action_size (int): The size of the action space.
        """
        super(Network, self).__init__()
        # Definition of the fully connected layers
        self.fc1 = nn.Linear(state_size, 256)  # Adjusted to the correct state_size
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
    state  = env.reset()
    total_reward = 0
    while True:
      action = agent.act(state)
      state, reward, done, _ = env.step(action[0])
      total_reward += reward
      if done:
        break
    episodes_rewards.append(total_reward)
  return episodes_rewards

def make_env():
  return SmartHomeTempControlEnv(start_from_random_day=True)

class EnvBatch:
    """
    A class to manage a batch of environment instances.

    This class facilitates the simultaneous stepping and resetting of multiple environment instances,
    allowing for efficient data collection from parallel interactions. This is especially useful
    in algorithms that leverage multiple workers or agents exploring in parallel to speed up learning.
    """

    def __init__(self, n_envs=number_environments):
        """
        Initializes the EnvBatch with a specified number of environment instances.

        Parameters:
            n_envs (int): The number of environment instances to manage.
        """
        # Creates a list of environment instances by calling make_env() for each one.
        self.envs = [make_env() for _ in range(n_envs)]

    def reset(self):
        """
        Resets all environments in the batch and returns their initial states.

        Returns:
            np.array: An array containing the initial state of each environment in the batch.
        """
        _states = []  # Initialize an empty list to hold the initial states of all environments.
        for env in self.envs:
            # Reset each environment to its initial state and append the state to the list.
            _states.append(env.reset())
        # Convert the list of states to a NumPy array for efficient processing and return it.
        return np.array(_states)

    def step(self, actions):
        """
        Performs a step in each environment using the given actions and returns the results.

        Parameters:
            actions (list or np.array): A list or array of actions, one for each environment in the batch.

        Returns:
            tuple: A tuple containing arrays of next states, rewards, done flags, and info objects for each environment.
        """
        # Step through each environment using the corresponding action. The zip function pairs each environment
        # with its action, and the step function is called on each pair, producing a list of results.
        next_states, rewards, dones, infos = map(np.array, zip(*[env.step(a) for env, a in zip(self.envs, actions)]))

        # Check if any environment has finished (done == True). If so, reset that environment and use its
        # new initial state as the next state, ensuring continuous interaction without manual resetting.
        for i in range(len(self.envs)):
            if dones[i]:
                next_states[i] = self.envs[i].reset()

        # Return the results as a tuple of arrays: next states, rewards, done flags, and info objects.
        # This format facilitates batch processing and integration with learning algorithms.
        return next_states, rewards, dones, infos


env = make_env()
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
print('State size: ', state_size)
print('Action size: ', action_size)

env_batch = EnvBatch(number_environments)
batch_states = env_batch.reset()
agent = A3C(state_size, action_size)

print("Batch states: ")
print(batch_states)
print(batch_states.shape)

with tqdm.trange(0, 3001) as progress_bar:
  for i in progress_bar:
    print("...")
    batch_actions = agent.act(batch_states)
    batch_next_states, batch_rewards, batch_dones, _ = env_batch.step(batch_actions)
    batch_rewards *= 0.01
    agent.step(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones)
    batch_states = batch_next_states
    if i % 1000 == 0:
      print("Average agent reward: ", np.mean(evaluate(agent, env, n_episodes = 10)))
