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
        return cls._instance

    def reset(self):
        self.all_scores.clear()
        self.iter = 0

    def get_avg_rewards(self, average_over=30):
        if not self.all_scores:
            return 0, 0
        average_over = min(len(self.all_scores), average_over)
        return sum(self.all_scores[-average_over:]) / average_over, average_over

    def log_reward(self, reward, name='Episode', average_over=30):
        self.all_scores.append(reward)

        average_of_episodes, number_of_episodes = self.get_avg_rewards(average_over)

        self.iter += 1
        if self.iter % 10 == 0:
            print(f'\r{name}: {self.iter}\tAverage Score Of Last {number_of_episodes} {name}s: {average_of_episodes:.2f}', end="")

    def store_loss(self, loss):
        # print(f'\rLoss: {loss:.5f}', end="")
        # TODO?
        pass

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
        plt.title('Scores over Environment Episodes')
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
