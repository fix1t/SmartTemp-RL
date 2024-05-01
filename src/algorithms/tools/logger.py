"""
    File: logger.py
    Author: Gabriel Biel

    Description: Logger singleton class to log the progress of the training
    and store the training information.
"""

import math
import os
import matplotlib.pyplot as plt
from env.environment import TempRegulationEnv
import torch
import yaml

import tools.plotter as plotter
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
        self.iter += 1
        if self.iter % 10 == 0:
            if self.iter % 100 == 0:
                average_of_episodes, number_of_episodes = self.get_avg_rewards(100)
                print(f'\r{name}: {self.iter}\tAverage Score Of Last 100 {name}s: {average_of_episodes:.2f}')
            else:
                average_of_episodes, number_of_episodes = self.get_avg_rewards(average_over)
                print(f'\r{name}: {self.iter}\tAverage Score Of Last {number_of_episodes} {name}s: {average_of_episodes:.2f}', end="")

    def store_loss(self, loss):
        # print(f'\rLoss: {loss:.5f}', end="")
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

    def save_agent_info(self, folder, agent, config, elapsed_time, extra_text=""):
        if not os.path.exists(folder):
            os.makedirs(folder)

        with open(f"{folder}/info.txt", "w") as f:
            f.write(f"---------------Training summary---------------------\n\n")

            if self.all_scores.__len__() == 0:
                f.write("No episodes were run\n")
                return


            max_reward = float(max(self.all_scores))
            average_reward = float(self.get_last_avg_score())

            summary = {
                "Elapsed Time (s)": math.trunc(elapsed_time),
                "Max reward per episode": math.trunc(max_reward),
                "Average over last 10 episodes": math.trunc(average_reward),
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
            f.write("\n\n")
            f.write(extra_text)

    @staticmethod
    def plot_all_in_one(agent, folder='out/plots', seed = 42,
                        atypical=False, winter=True, name=None):

        atypical_config ='env/environment_configuration_atypic_enhanced.json'
        typical_config = 'env/environment_configuration.json'

        winter_data = 'data/feb_2020.csv'
        summer_data = 'data/june_2020.csv'

        env_configuration_path = atypical_config if atypical else typical_config
        temp_data = winter_data if winter else summer_data

        env = TempRegulationEnv(
            start_from_random_day=False,
            seed=int(seed),
            max_steps_per_episode=4*24*7,
            config_file=env_configuration_path,
            temp_data_file=temp_data
        )

        # Simulate the agent in the environment for a week
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = agent.get_action(obs)
            obs, _, done, _, _ = env.step(action)

        if not os.path.exists(folder):
            os.makedirs(folder)

        time, indoor_temp, outdoor_temp = env.get_temperature_data()
        _, heating = env.get_heating_data()
        _, occupancy = env.get_occupancy_data()
        occupancy = occupancy['father']

        target_length = min(len(outdoor_temp), len(indoor_temp), len(occupancy), len(heating), len(time))

        # Truncate arrays to the shortest length among them
        outdoor_temp = outdoor_temp[:target_length]
        indoor_temp = indoor_temp[:target_length]
        occupancy = occupancy[:target_length]
        heating = heating[:target_length]
        time = time[:target_length]

        file_name = 'typical_aio' if not atypical else 'atypical_aio'
        if name is not None:
            file_name = name
        plotter.plot_all_in_one(outdoor_temp, indoor_temp, occupancy,
                                heating, time, output_dir=folder, name=file_name)

    def save_all_aio_plots(self, agent, folder='out/plots'):
        self.plot_all_in_one(agent, f"{folder}", winter=False, seed=42,  atypical=False, name='summer')
        self.plot_all_in_one(agent, f"{folder}", winter=True, seed=42,  atypical=False, name='winter')
        self.plot_all_in_one(agent, f"{folder}", winter=True, seed=22, atypical=True, name='atypical')
