import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from agent.replay_memory import ReplayMemory  # Custom module for replay memory
from agent.network import Network  # Custom module for the neural network

# Hyperparameters for the DQL agent
learning_rate = 5e-4  # Learning rate for the optimizer
minibatch_size = 100  # Size of the minibatch from replay memory for learning
discount_factor = 0.99  # Discount factor for future rewards
replay_buffer_size = int(1e5)  # Size of the replay buffer
interpolation_parameter = 1e-3  # Used in soft update of target network

class DQL():
    def __init__(self, state_size, action_size):
        """
        Initializes the agent.

        Parameters:
            state_size (int): Dimension of each state.
            action_size (int): Number of actions available to the agent.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Device configuration
        self.state_size = state_size  # State space size
        self.action_size = action_size  # Action space size

        # Q-Networks
        self.local_qnetwork = Network(state_size, action_size).to(self.device)  # Local network for learning
        self.target_qnetwork = Network(state_size, action_size).to(self.device)  # Target network for stable Q-targets

        # Optimizer
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=learning_rate)

        # Replay memory
        self.memory = ReplayMemory(replay_buffer_size)  # Replay memory to store experiences
        self.t_step = 0  # Counter to track steps for updating

    def step(self, state, action, reward, next_state, done):
        """
        Stores experience in replay memory and learns every 4 steps.

        Parameters:
            state (array_like): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (array_like): The next state.
            done (bool): Whether the episode has ended.
        """
        # Save experience in replay memory
        self.memory.push((state, action, reward, next_state, done))

        # Learn every 4 steps
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0 and len(self.memory.memory) > minibatch_size:
            experiences = self.memory.sample(minibatch_size)
            self.learn(experiences, discount_factor)

    def act(self, state, epsilon=0.):
        """
        Returns actions for given state following the current policy.

        Parameters:
            state (array_like): Current state.
            epsilon (float): Epsilon, for epsilon-greedy action selection.

        Returns:
            int: The action selected.
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_qnetwork.eval()  # Set network to evaluation mode
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
        self.local_qnetwork.train()  # Set network back to train mode

        # Epsilon-greedy action selection
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())  # Exploit
        else:
            return random.choice(np.arange(self.action_size))  # Explore

    def learn(self, experiences, discount_factor):
        """
        Updates value parameters using given batch of experience tuples.

        Parameters:
            experiences (Tuple[torch.Variable]): Tuple of (s, a, r, s', done) tuples.
            discount_factor (float): Discount factor for future rewards.
        """
        states, next_states, actions, rewards, dones = experiences

        # Get max predicted Q values (for next states) from target model
        next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        q_targets = rewards + (discount_factor * next_q_targets * (1 - dones))

        # Get expected Q values from local model
        q_expected = self.local_qnetwork(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(q_expected, q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network
        self.soft_update(self.local_qnetwork, self.target_qnetwork, interpolation_parameter)

    def soft_update(self, local_model, target_model, interpolation_parameter):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Parameters:
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            interpolation_parameter (float): interpolation parameter τ
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(interpolation_parameter * local_param.data + (1.0 - interpolation_parameter) * target_param.data)
