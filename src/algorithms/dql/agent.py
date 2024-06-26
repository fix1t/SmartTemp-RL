"""
    File: dql/agent.py
    Author: Gabriel Biel

    Description: Agent class for the Deep Q-Learning (DQL) algorithm.
"""

import random
from time import sleep
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from algorithms.dql.memory import ReplayMemory
from algorithms.tools.logger import Logger

class Agent():
    def __init__(self, local_qnetwork, target_qnetwork, env, **hyperparameters):
        """
        Initializes the agent.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Device configuration

        self.init_hyperparameters(hyperparameters)

        self.env = env

        self.local_qnetwork = local_qnetwork.to(self.device)
        self.target_qnetwork = target_qnetwork.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=self.learning_rate)

        # Replay memory
        self.memory = ReplayMemory(self.replay_buffer_size)  # Replay memory to store experiences
        self.t_step = 1  # Counter to track steps for updating

    def init_hyperparameters(self, hyperparameters):
        self.learning_rate = hyperparameters.get('learning_rate', 5e-4)
        self.batch_size = hyperparameters.get('batch_size', 100)
        self.discount_factor = hyperparameters.get('discount_factor', 0.99)
        self.replay_buffer_size = hyperparameters.get('replay_buffer_size', int(1e5))
        self.interpolation_parameter = hyperparameters.get('interpolation_parameter', 1e-3)
        self.learning_freqency = hyperparameters.get('learning_freqency', 4)

        hyperparameters = {
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'discount_factor': self.discount_factor,
            'replay_buffer_size': self.replay_buffer_size,
            'interpolation_parameter': self.interpolation_parameter,
            'learning_freqency': self.learning_freqency
        }

        self.epsilon_starting_value  = 1.0
        self.epsilon_ending_value  = 0.01
        self.epsilon_decay_value  = 0.995
        self.epsilon = self.epsilon_starting_value

        print("--------------------------------")
        print(f"DQL agent loaded with hyperparameters:\n{hyperparameters}")


    def step(self, state, action, reward, next_state, done):
        """
        Stores experience in replay memory and learns every N steps.

        Parameters:
            state (array): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (array): The next state.
            done (bool): Whether the episode has ended.
        """
        # Save experience in replay memory
        self.memory.push((state, action, reward, next_state, done))

        self.t_step = (self.t_step + 1) % self.learning_freqency

        if self.t_step == 0 and len(self.memory.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.update_policy(experiences, self.discount_factor)

    def get_action(self, state, epsilon=0.):
        """
        Returns actions for given state following the current policy.

        Parameters:
            state (array): Current state.
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
            action=  np.argmax(action_values.cpu().data.numpy()) # Exploit
        else:
            action=  random.choice(np.arange(self.env.action_space.n)) # Explore
        return action, None # None is for compatibility with other agents (PPO returns the log probability of the action taken)

    def update_policy(self, experiences, discount_factor):
        """
        Updates the policy network (Q-network) parameters using a given batch of experience tuples.
        This method implements the core of the DQL algorithm, optimizing the network to better predict
        Q-values given states and actions.

        Parameters:
            experiences (Tuple[torch.Variable]): Batch of experiences, where each experience is a tuple
                                                of (state, action, reward, next_state, done) tensors.
            discount_factor (float): Gamma (γ), the discount factor used to weigh future rewards.

        This method performs the following steps to update the Q-network:
        1. Calculate the target Q-values for the next states (s') using the target network, taking the max
        Q-value for each next state. This is done to decouple the selection of action from the evaluation
        to mitigate positive feedback loops.
        2. Compute the Q-targets for the current states (s) by applying the Bellman equation, incorporating
        rewards and discounted future rewards.
        3. Obtain the expected Q-values from the local (policy) network for the actions taken.
        4. Calculate the loss between the expected Q-values and the Q-targets using mean squared error (MSE),
        which represents the temporal difference error.
        5. Backpropagate the loss to update the weights of the local Q-network.
        6. Periodically, the target Q-network is softly updated with weights from the local Q-network to slowly
        track the learned value function, improving stability.
        """
        states, next_states, actions, rewards, dones = experiences

        # Compute Q values for next states from the target network and calculate Q targets
        next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (discount_factor * next_q_targets * (1 - dones))

        # Get expected Q values from the local network for the current actions
        q_expected = self.local_qnetwork(states).gather(1, actions)

        # Compute and backpropagate loss
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the weights of the target network to slowly track the learned Q-values
        self._soft_update_network(self.local_qnetwork, self.target_qnetwork, self.interpolation_parameter)

    def _soft_update_network(self, local_model, target_model, interpolation_parameter):
        """
        Performs a soft update on the target network's parameters. This method blends the parameters
        of the local Q-network and the target Q-network using an interpolation parameter, τ (tau), to
        slowly update the target network. This approach, known as Polyak averaging, helps maintain
        stability in learning by ensuring that the target values change slowly over time, reducing
        the risk of divergence.

        Parameters:
            local_model (PyTorch model): The local (policy) network, from which weights are copied.
            target_model (PyTorch model): The target network, to which weights are updated.
            interpolation_parameter (float): The interpolation parameter τ (tau), typically a small
                                            value (e.g., 0.001), controlling the extent to which the
                                            target network is updated.

        The update formula is as follows:
        θ_target = τ*θ_local + (1 - τ)*θ_target
        This formula is applied to each parameter of the target network, ensuring that the update
        is gradual and keeps the target network's output stable, which is crucial for the convergence
        of the DQL algorithm.
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(interpolation_parameter * local_param.data + (1.0 - interpolation_parameter) * target_param.data)

    def _periodic_update_target_network(self):
        """
        Periodically updates the target network with the weights from the local network.
        This method is called every few steps (as specified by the learning_freqency parameter)
        to ensure that the target network tracks the learned Q-values from the local network.
        """
        self.t_step = (1 + self.t_step) % self.learning_freqency
        if self.t_step == 0:
            self.target_qnetwork.load_state_dict(self.local_qnetwork.state_dict())

    def train(self, total_timesteps=4*24*365*10):
        # For retraining the agent, motivate to explore more - learn possibly more
        if self.epsilon <= self.epsilon_ending_value:
            self.epsilon = 5 * self.epsilon_ending_value

        t_so_far = 0
        while t_so_far < total_timesteps:
            state, _ = self.env.reset(start_from_random_day=False)
            acc_reward = 0
            done = False
            while not done:
                action, _ = self.get_action(state, self.epsilon)
                next_state, reward, done, _, _ = self.env.step(action)
                self.step(state, action, reward, next_state, done)
                state = next_state
                acc_reward += reward
                t_so_far += 1

            self.epsilon = max(self.epsilon_ending_value, self.epsilon_decay_value * self.epsilon)
            Logger().log_reward(acc_reward)

    def load_local_qnetwork(self, path):
        self.local_qnetwork.load_state_dict(torch.load(path))

    def load_target_qnetwork(self, path):
        self.target_qnetwork.load_state_dict(torch.load(path))

    def test_policy(self, total_timesteps=4*24*180, render=True):
        t_so_far = 0
        self.env.set_max_steps_per_episode(total_timesteps)
        obs, _ = self.env.reset()
        done = False
        while not done:
            if render:
                self.env.render()
            action, _ = self.get_action(obs)
            obs, _, done, _, _ = self.env.step(action)
            t_so_far += 1
            sleep(0.001)

        print(f"Total timesteps: {t_so_far}")
        print("Done testing.")
        print("--------------------------------")
        sleep(5*60)
        self.env.close()
