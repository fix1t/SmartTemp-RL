import os
import matplotlib.pyplot as plt
import torch
import yaml

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

    def reset(self):
        self.all_scores = []
        self.iter = 0
        self.dql = True

    def log_episode(self, score, episode=None):
        self.all_scores.append(score)
        if episode is not None:
            self.iter = episode
        else:
            self.iter += 1
        num_scores = min(10, len(self.all_scores))
        average_score_of_last_10_episodes = sum(self.all_scores[-num_scores:]) / num_scores
        print(f'\rEpisode: {self.iter}\tAverage Score Of Last {num_scores}: {average_score_of_last_10_episodes:.2f}', end="")

    def log_iteration(self, score, iter=None):
        self.dql = False
        self.all_scores.append(score)
        if iter is not None:
            self.iter = iter
        else:
            self.iter += 1
        num_scores = min(10, len(self.all_scores))
        average_score_of_last_10_episodes = sum(self.all_scores[-num_scores:]) / num_scores
        print(f'\rIteration: {iter}\tAverage Score Of Last {num_scores}: {average_score_of_last_10_episodes:.2f}', end="")

    @staticmethod
    def save_trained_agent(agent, folder='out/agents'):
        if not os.path.exists(folder):
            os.makedirs(folder)

        if hasattr(agent, 'actor'):
            torch.save(agent.actor.state_dict(), f"{folder}/actor_model.pth")
        if hasattr(agent, 'critic'):
            torch.save(agent.critic.state_dict(), f"{folder}/critic_model.pth")

        if hasattr(agent, 'local_qnetwork'):
            torch.save(agent.local_qnetwork.state_dict(), f"{folder}/local_qnetwork.pth")
        if hasattr(agent, 'target_qnetwork'):
            torch.save(agent.target_qnetwork.state_dict(), f"{folder}/target_qnetwork.pth")

    def plot_scores(self, folder='out/plots'):
        if not os.path.exists(folder):
            os.makedirs(folder)

        file_full_path = f'{folder}/score.png'

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
        plt.savefig(file_full_path)
        plt.close()

    def get_last_avg_score(self, num_scores=10):
        num_scores = min(num_scores, len(self.all_scores))
        if num_scores == 0:
            return 0
        return sum(self.all_scores[-num_scores:]) / num_scores

    def save_agent_info(self, folder, agent, config, elapsed_time):
        if not os.path.exists(folder):
            os.makedirs(folder)

        with open(f"{folder}/info.txt", "w") as f:
            f.write(f"---------------Training summary---------------------\n\n")

            if self.all_scores.__len__() == 0:
                f.write("No episodes were run\n")
                return

            summary = {
                "Elapsed Time": elapsed_time,
                "Max reward per episode": max(self.all_scores),
                "Average over last 10 episodes": self.get_last_avg_score(),
                "Total episodes/iterations": len(self.all_scores),
                "Total steps": len(self.all_scores) * agent.env.max_steps_per_episode,
            }
            f.write(yaml.dump(summary))
            f.write("\n\n")

            f.write(f"---------------Agent's configuration----------------\n\n")
            f.write(yaml.dump(config))
            f.write("\n\n")

            f.write(f"---------------Agent's network----------------------\n\n")
            if hasattr(agent, 'actor'):
                f.write(f"Actor network:\n")
                f.write(f"{agent.actor}\n\n")
                f.write(f"Critic network:\n")
                f.write(f"{agent.critic}\n\n")

            else:
                f.write(f"Local Q-network:\n")
                f.write(f"{agent.local_qnetwork}\n\n")
                f.write(f"Target Q-network:\n")
                f.write(f"{agent.target_qnetwork}\n")
