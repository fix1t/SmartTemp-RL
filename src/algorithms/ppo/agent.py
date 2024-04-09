import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from algorithms.tools.logger import Logger
import time

class Agent:
    # Initial setup of the agent with actor-critic networks and environment
    def __init__(self, actor_network, critic_network, env, **hyperparameters):
        self.env = env
        self.actor = actor_network
        self.critic = critic_network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.seed = hyperparameters.get('seed', None)
        if self.seed:
            torch.manual_seed(self.seed)

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
            t_so_far += np.sum(batch_data[-1])
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
        batch_rtgs = self.compute_rtgs(batch_rews)
        batch_data = self._convert_data_to_tensor(batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens)

        self._log_summary(batch_rews)

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
        action_probs = self.actor(torch.tensor(obs, dtype=torch.float).unsqueeze(0))
        dist = torch.distributions.Categorical(probs=action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    # Updates the policy based on collected batch data
    def update_policy(self, batch_data):
        batch_obs, batch_acts, batch_log_probs, batch_rtgs, _ = batch_data
        for _ in range(self.n_updates_per_iteration):
            V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            self._update_networks(A_k, batch_log_probs, curr_log_probs, V, batch_rtgs)

    # Evaluates the current policy and value function
    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs)
        action_probs = self.actor(batch_obs)
        dist = torch.distributions.Categorical(probs=action_probs)
        log_probs = dist.log_prob(batch_acts)
        return V.squeeze(), log_probs

    # Performs the actual network updates
    def _update_networks(self, A_k, batch_log_probs, curr_log_probs, V, batch_rtgs):
        ratios = torch.exp(curr_log_probs - batch_log_probs)
        surr1 = ratios * A_k
        surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
        actor_loss = (-torch.min(surr1, surr2)).mean()
        critic_loss = nn.MSELoss()(V, batch_rtgs)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        self.logger.store_loss(actor_loss.detach().item())

    # Logs summary of training progress
    def _log_summary(self, batch_rews):
        # This computes the mean of episode totals, which seems to be your intention
        avg_ep_rews = np.mean([np.sum(ep) for ep in batch_rews])
        self.logger.log_iteration(avg_ep_rews)

    def load_actor(self, path):
        self.actor.load_state_dict(torch.load(path))

    def load_critic(self, path):
        self.critic.load_state_dict(torch.load(path))

    def test_policy(self, total_timesteps=4*24*180, render=True):
        print("-------Testing PPO agent-------")
        print(f"Testing for {total_timesteps} timesteps.")
        print("--------------------------------")
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
