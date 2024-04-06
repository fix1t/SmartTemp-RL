from collections import deque
import os
from datetime import datetime
import matplotlib.pyplot as plt
import torch

class Logger():
    """
    Logger singleton class to log the progress of the training.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance.all_scores = []
            cls._instance.iter = 0
            cls._instance.dql = True
        return cls._instance

    def log_episode(self, score, episode=None):
        self.all_scores.append(score)
        if episode is not None:
            self.iter = episode
        else:
            self.iter += 1
        num_scores = min(10, len(self.all_scores))
        average_score_of_last_10_episodes = sum(self.all_scores[-10:]) / num_scores
        print(f'\rEpisode: {self.iter}\tAverage Score Of Last 10: {average_score_of_last_10_episodes:.2f}', end="")

    def log_iteration(self, score, iter=None):
        self.dql = False
        self.all_scores.append(score)
        if iter is not None:
            self.iter = iter
        else:
            self.iter += 1
        num_scores = min(10, len(self.all_scores))
        average_score_of_last_10_episodes = sum(self.all_scores[-10:]) / num_scores
        print(f'\rIteration: {iter}\tAverage Score Of Last 10: {average_score_of_last_10_episodes:.2f}', end="")

    @staticmethod
    def save_agent(agent, folder='out/agents'):
        if not os.path.exists(folder):
            os.makedirs(folder)
        current_time = datetime.now().strftime("%m-%d_%H-%M")
        filename = f'{folder}/{current_time}'

        if hasattr(agent, 'actor'):
            torch.save(agent.actor.state_dict(), f"{filename}_actor_model.pth")
        if hasattr(agent, 'critic'):
            torch.save(agent.critic.state_dict(), f"{filename}_critic_model.pth")

        if hasattr(agent, 'local_qnetwork'):
            torch.save(agent.local_qnetwork.state_dict(), f"{filename}_local_qnetwork.pth")
        if hasattr(agent, 'target_qnetwork'):
            torch.save(agent.target_qnetwork.state_dict(), f"{filename}_target_qnetwork.pth")

    def plot_scores(self, folder='out/plots', filename=None):
        if not os.path.exists(folder):
            os.makedirs(folder)
        if filename is None:
            current_time = datetime.now().strftime("%m-%d_%H-%M")
            filename = f'{folder}/{current_time}.png'

        plt.figure(figsize=(20, 10))
        plt.plot(self.all_scores)
        if self.dql:
            plt.title('Scores over Episodes')
            plt.xlabel('Episode')
        else:
            plt.title('Scores over Iterations')
            plt.xlabel('Iteration')
        plt.title('Scores over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.savefig(filename)
        plt.close()

