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
        self.t_step = 0  # Counter to track steps for updating

    def init_hyperparameters(self, hyperparameters):
        self.learning_rate = hyperparameters.get('learning_rate', 5e-4)
        self.batch_size = hyperparameters.get('batch_size', 100)
        self.discount_factor = hyperparameters.get('discount_factor', 0.99)
        self.replay_buffer_size = hyperparameters.get('replay_buffer_size', int(1e5))
        self.interpolation_parameter = hyperparameters.get('interpolation_parameter', 1e-3)
        self.target_update_frequency = hyperparameters.get('target_update_frequency', 100)
        self.local_update_frequency = hyperparameters.get('local_update_frequency', self.target_update_frequency//2)

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

        if len(self.memory.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.t_step = (self.t_step + 1) % self.target_update_frequency

            if self.t_step % self.local_update_frequency == 0:
                self.update_policy(experiences, self.discount_factor)

            if self.t_step == 0:
                self.soft_update_target_network()

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
            return np.argmax(action_values.cpu().data.numpy())  # Exploit
        else:
            return random.choice(np.arange(self.env.action_space.n))  # Explore

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

    def soft_update_target_network(self):
        """
        Performs a soft update on the target network's parameters. This method blends the parameters
        of the local Q-network and the target Q-network using an interpolation parameter, τ (tau), to
        slowly update the target network. This approach, known as Polyak averaging, helps maintain
        stability in learning by ensuring that the target values change slowly over time, reducing
        the risk of divergence.

        The update formula is as follows:
        θ_target = τ*θ_local + (1 - τ)*θ_target
        This formula is applied to each parameter of the target network, ensuring that the update
        is gradual and keeps the target network's output stable, which is crucial for the convergence
        of the DQL algorithm.
        """
        for target_param, local_param in zip(self.target_qnetwork.parameters(), self.local_qnetwork.parameters()):
            target_param.data.copy_(self.interpolation_parameter * local_param.data + (1.0 - self.interpolation_parameter) * target_param.data)

    def train(self, total_timesteps=4*24*365*10):
        epsilon_starting_value  = 1.0
        epsilon_ending_value  = 0.01
        epsilon_decay_value  = 0.995
        epsilon = epsilon_starting_value

        t_so_far = 0
        while t_so_far < total_timesteps:
            state, _ = self.env.reset(start_from_random_day=False)
            acc_reward = 0
            done = False
            while not done:
                action = self.get_action(state, epsilon)
                next_state, reward, done, _, _ = self.env.step(action)
                self.step(state, action, reward, next_state, done)
                state = next_state
                acc_reward += reward
                t_so_far += 1

            epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)
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
            action = self.get_action(obs)
            obs, _, done, _, _ = self.env.step(action)
            t_so_far += 1
            sleep(0.01)

        print(f"Total timesteps: {t_so_far}")
        print("Done testing.")
        print("--------------------------------")
        sleep(5*60)
        self.env.close()
