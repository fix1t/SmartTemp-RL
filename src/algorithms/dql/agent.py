from collections import deque
import random
from time import sleep
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from algorithms.dql.network import Network
from algorithms.dql.memory import ReplayMemory
from algorithms.tools.logger import Logger

class Agent():
    def __init__(self, policy_class, env, **hyperparameters):
        """
        Initializes the agent.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Device configuration

        self.init_hyperparameters(hyperparameters)

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n # Discrete action space

        # Q-Networks
        self.local_qnetwork = policy_class(self.obs_dim, self.act_dim).to(self.device)  # Local network for learning
        self.target_qnetwork = policy_class(self.obs_dim, self.act_dim).to(self.device)  # Target network for stable Q-targets

        # Optimizer
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=self.learning_rate)

        # Replay memory
        self.memory = ReplayMemory(self.replay_buffer_size)  # Replay memory to store experiences
        self.t_step = 0  # Counter to track steps for updating

    def init_hyperparameters(self, hyperparameters):
        """
        Initializes hyperparameters for the agent.
        """
        learning_rate = 5e-4  # Learning rate for the optimizer
        minibatch_size = 100  # Size of the minibatch from replay memory for learning
        discount_factor = 0.99  # Discount factor for future rewards
        replay_buffer_size = int(1e5)  # Size of the replay buffer
        interpolation_parameter = 1e-3  # Used in soft update of target network
        self.learning_rate = hyperparameters.get('learning_rate', learning_rate)
        self.minibatch_size = hyperparameters.get('minibatch_size', minibatch_size)
        self.discount_factor = hyperparameters.get('discount_factor', discount_factor)
        self.replay_buffer_size = hyperparameters.get('replay_buffer_size', replay_buffer_size)
        self.interpolation_parameter = hyperparameters.get('interpolation_parameter', interpolation_parameter)

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
        NUMBERS_OF_STEPS_BEFORE_LEARNING = 4
        self.t_step = (self.t_step + 1) % NUMBERS_OF_STEPS_BEFORE_LEARNING
        if self.t_step == 0 and len(self.memory.memory) > self.minibatch_size:
            experiences = self.memory.sample(self.minibatch_size)
            self.learn(experiences, self.discount_factor)

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
            return random.choice(np.arange(self.act_dim))  # Explore

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
        self.soft_update(self.local_qnetwork, self.target_qnetwork, self.interpolation_parameter)

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

    def train(self, total_timesteps=4*24*365*10):
        epsilon_starting_value  = 1.0
        epsilon_ending_value  = 0.01
        epsilon_decay_value  = 0.995
        epsilon = epsilon_starting_value

        number_episodes = total_timesteps // self.env.max_steps_per_episode
        print("-------Training DQL agent-------")
        print(f"Training for {total_timesteps} timesteps with {self.env.max_steps_per_episode} steps per episode.")
        print(f"Total number of episodes: {number_episodes}")
        print("--------------------------------")

        for episode in range(1, number_episodes+1):
            state, _ = self.env.reset(start_from_random_day=False)
            score = 0
            done = False
            while not done:
                action = self.act(state, epsilon)
                next_state, reward, done, _, _ = self.env.step(action)
                self.step(state, action, reward, next_state, done)
                state = next_state
                score += reward

            epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)

            Logger().log_episode(score, episode)

    def test_policy(self, total_timesteps=4*24*180, render=True):
        print("-------Testing DQL agent-------")
        print(f"Testing for {total_timesteps} timesteps = {total_timesteps//(4*24)} days.")
        print("--------------------------------")
        t_so_far = 0
        self.env.set_max_steps_per_episode(total_timesteps)
        obs, _ = self.env.reset()
        done = False
        while not done:
            if render:
                self.env.render()
            action = self.act(obs)
            obs, _, done, _, _ = self.env.step(action)
            t_so_far += 1
            sleep(0.01)

        print(f"Total timesteps: {t_so_far}")
        print("Done testing.")
        print("--------------------------------")
        sleep(5*60)
