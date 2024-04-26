"""
    File: ppo/agent.py
    Author: Gabriel Biel

    Description: This file contains the implementation of the PPO agent class. The agent
    is responsible for training the actor and critic networks using the Proximal Policy
    Optimization (PPO) algorithm.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from algorithms.tools.logger import Logger
import time

class Agent:
    # Initial setup of the agent with actor-critic networks and environment
    def __init__(self, actor_network, critic_network, env, **hyperparameters):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.actor = actor_network.to(self.device)
        self.critic = critic_network.to(self.device)
        self._init_hyperparameters(hyperparameters)
        self._init_optimizers()
        self._init_logger()

    # Initializes hyperparameters from arguments or sets defaults
    def _init_hyperparameters(self, hyperparameters):
        self.learning_rate = hyperparameters.get('learning_rate', 0.005)
        self.discount_factor = hyperparameters.get('discount_factor', 0.95)
        self.clip = hyperparameters.get('clip', 0.2)
        self.batch_size = hyperparameters.get('batch_size', 4800)
        self.n_updates_per_iteration = hyperparameters.get('n_updates_per_iteration', 5)

    # Initializes optimizers for both actor and critic networks
    def _init_optimizers(self):
        self.actor_optim = Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.learning_rate)

    # Initializes the logger for tracking progress
    def _init_logger(self):
        self.logger = Logger()
        self.logger.reset()

    # Main training loop
    def train(self, total_timesteps):
        t_so_far = 0
        i_so_far = 0
        while t_so_far < total_timesteps:
            batch_data = self.collect_batch_data()
            batch_lens = batch_data[-1]
            t_so_far += np.sum(batch_lens)
            i_so_far += 1
            self.update_policy(batch_data)

    # Collects data from environment interaction
    def collect_batch_data(self):
        batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens = [], [], [], [], []
        t = 0
        while t < self.batch_size:
            obs, _ = self.env.reset()
            action, log_prob, rew, done = None, None, 0, False
            ep_rews = []
            while not done:
                action, log_prob = self.get_action(obs)
                next_obs, rew, done, _, _ = self.env.step(action)
                ep_rews.append(rew)
                self._store_transition(batch_obs, batch_acts, batch_log_probs, obs, action, log_prob)
                obs = next_obs
                t += 1
            batch_rews.append(ep_rews)
            batch_lens.append(len(ep_rews))
            self._log_episode(batch_rews)

        batch_rtgs = self.compute_rtgs(batch_rews)
        batch_data = self._convert_data_to_tensor(batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens)

        return batch_data

    def _convert_data_to_tensor(self, batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens):
        batch_obs = torch.tensor(np.array(batch_obs, dtype=np.float32), dtype=torch.float, device=self.device)
        batch_acts = torch.tensor(batch_acts, dtype=torch.short, device=self.device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float, device=self.device)
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float, device=self.device)
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    # Stores transition data
    def _store_transition(self, batch_obs, batch_acts, batch_log_probs, obs, action, log_prob):
        batch_obs.append(obs)
        batch_acts.append(action)
        batch_log_probs.append(log_prob)

    # Computes rewards-to-go
    def compute_rtgs(self, batch_rews):
        batch_rtgs = []
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + self.discount_factor * discounted_reward
                batch_rtgs.insert(0, discounted_reward)
        return batch_rtgs

    # Selects an action based on the current policy
    def get_action(self, obs):
        # Assertion for device check before tensor operation
        action_probs = self.actor(torch.tensor(obs, dtype=torch.float, device=self.device).unsqueeze(0))
        dist = torch.distributions.Categorical(probs=action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update_policy(self, batch_data):
        """
        Updates the policy based on collected batch data. This function implements
        the core PPO policy update step. It iterates over the batch data multiple
        times (as defined by `n_updates_per_iteration`) to perform multiple epochs
        of optimization on the same batch of data, which is a key feature of PPO
        to stabilize training.

        Args:
            batch_data: A tuple containing batch observations, actions, log probabilities,
                        rewards-to-go (RTGs), and episode lengths. This data is used to
                        calculate advantages and update the policy and value networks.

        The update process involves calculating the advantage (A_k) by subtracting the value
        estimates (V) from the RTGs. The advantage is then standardized to reduce variance,
        following standard practice in advantage-based policy optimization methods.

        The `_update_networks` method is then called to perform the actual optimization step,
        adjusting the actor (policy) and critic (value function) networks based on the computed
        advantages, old and current log probabilities, and value estimates.
        """
        batch_obs, batch_acts, batch_log_probs, batch_rtgs, _ = batch_data
        for _ in range(self.n_updates_per_iteration):
            # Calculate advantage at k-th iteration
            V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V.detach()  # Detach V to stop gradients

            # Trick found in https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-2-4-f9d8b8aa938a article
            #    "Trick I use that isn't in the pseudocode. Normalizing advantages
            #    isn't theoretically necessary, but in practice it decreases the variance of
            #    our advantages and makes convergence much more stable and faster. I added this because
            #    solving some environments was too unstable without it."
            # And it works well for me too.

            # + 1e-10 is a constant only to prevent division by zero
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            self._update_networks(A_k, batch_log_probs, curr_log_probs, V, batch_rtgs)


    def evaluate(self, batch_obs, batch_acts):
        """
        Evaluates the current policy and value function for a given batch of observations
        and actions. This method is crucial for PPO's actor-critic architecture, providing
        the necessary information to compute advantages and update the networks.

        Args:
            batch_obs: Tensor containing a batch of observations.
            batch_acts: Tensor containing a batch of actions taken by the policy.

        Returns:
            V: The value function estimates for the given observations, representing
            the expected return. Used in advantage calculation and critic update.
            log_probs: The log probabilities of taking the actions in `batch_acts`
                    under the current policy. Used to compute the probability ratio
                    for the PPO objective.

        This function uses the current policy (actor) to compute the probability
        distribution over actions given the observations, then calculates the log
        probabilities of the actual actions taken. It also computes the value function
        estimates for the observations using the critic network.
        """
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        action_probs = self.actor(batch_obs)
        dist = torch.distributions.Categorical(probs=action_probs)
        log_probs = dist.log_prob(batch_acts)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs

    def _update_networks(self, A_k, batch_log_probs, curr_log_probs, V, batch_rtgs):
        """
        Performs the actual network updates for both the actor (policy) and critic
        (value function) using the computed advantages, old and current log probabilities,
        and value estimates.

        Args:
            A_k: The standardized advantages for the batch of data.
            batch_log_probs: The log probabilities of the actions in the batch
                            under the policy used to generate the data.
            curr_log_probs: The current log probabilities of the batch actions under
                            the updated policy.
            V: The value function estimates for the batch observations.
            batch_rtgs: The rewards-to-go for the batch, used in the critic loss.

        This method calculates the PPO clipped objective for the actor loss, which minimizes
        the difference between the old and new policy, constrained by a clipping factor to
        prevent too large policy updates. The critic loss is computed as the mean squared error
        between the predicted values (V) and the actual returns (batch_rtgs), aiming to improve
        the value function estimation.

        The method concludes by performing gradient descent to update both networks, using
        separate optimizers for the actor and critic. It also logs the actor loss for monitoring.
        """
        # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
        # NOTE: we just subtract the logs, which is the same as
        # dividing the values and then canceling the log with e^log.
        ratios = torch.exp(curr_log_probs - batch_log_probs)  # Probability ratio for actions

        # Calculate surrogate losses
        surr1 = ratios * A_k  # Unclipped objective
        surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k  # Clipped objective

        # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
        # the performance function, but Adam minimizes the loss. So minimizing the negative
        # performance function maximizes it.
        actor_loss = (-torch.min(surr1, surr2)).mean()  # PPO's actor objective
        critic_loss = nn.MSELoss()(V, batch_rtgs)  # Critic loss

        # Calculate gradients and perform backward propagation for actor network
        self.actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optim.step()

        # Calculate gradients and perform backward propagation for critic networ
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Log actor loss
        self.logger.store_loss(actor_loss.detach().item())

    # Logs summary of training progress
    def _log_episode(self, batch_rews):
        reward = np.mean([np.sum(ep_rews) for ep_rews in batch_rews])
        self.logger.log_reward(reward, name='Episode')

    def load_actor(self, path):
        self.actor.load_state_dict(torch.load(path, map_location=self.device))

    def load_critic(self, path):
        self.critic.load_state_dict(torch.load(path, map_location=self.device))

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
            time.sleep(0.01)
        print(f"Total timesteps: {t_so_far}")
        print("Done testing.")
        print("--------------------------------")
        time.sleep(5*60)
        self.env.close()
