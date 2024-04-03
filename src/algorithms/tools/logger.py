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
            cls._instance.average_score = []  # This stores the history of average scores.
            cls._instance._episode_num = 0
        return cls._instance

    def log(self, score, episode_num=None):
        self.average_score.append(score)  # Corrected to append to the class attribute

        if episode_num is not None:
            self._episode_num = episode_num
        else:
            self._episode_num += 1

        self.print_progress(score, self._episode_num)  # Pass score correctly

    @staticmethod
    def print_progress(score, episode_num=None):
        if episode_num is None:
            print(f'\rAverage Score: {score:.2f}', end="")
        else:
            print(f'\rEpisode {episode_num}\tAverage Score: {score:.2f}', end="")

    @staticmethod
    def save_agent(agent, folder='out/agents'):
        if not os.path.exists(folder):
            os.makedirs(folder)
        current_time = datetime.now().strftime("%m-%d_%H-%M")
        filename = f'{folder}/{current_time}.pth'
        torch.save(agent.local_qnetwork.state_dict(), filename)

    def plot_scores(self, filename=None):
        if not os.path.exists('out/plots'):
            os.makedirs('out/plots')
        if filename is None:
            current_time = datetime.now().strftime("%m-%d_%H-%M")
            filename = f'out/plots/{current_time}.png'
        plt.figure(figsize=(10, 5))
        plt.plot(self.average_score)
        plt.title('Scores over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.savefig(filename)
        plt.close()

