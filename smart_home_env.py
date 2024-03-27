import gym
from gym import spaces
import numpy as np
from datetime import datetime

from modules.occupancy_manager import OccupancyManager
from modules.temperature_manager import TemperatureManager
from modules.configuration_manager import ConfigurationManager
from modules.time_manager import TimeManager
from modules.heating_system import HeatingSystem
from tools.renderer import SimulationRenderer
from tools.csv_line_reader import CSVLineReader

class SmartHomeTempControlEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, start_from_random_day=True, run_for_days=7):
        super(SmartHomeTempControlEnv, self).__init__()
        self.action_space = spaces.Discrete(2)  # 0: Heat Up, 1: Do Nothing

        #TODO: Define observation space - Current temperature, outside temperature, occupancy, heating system energy
        self.observation_space = spaces.Box(low=np.array([0, -30,0]), high=np.array([30, 40, 5]), dtype=np.float32)
        self.total_reward = 0
        self.run_for_days = run_for_days
        self.reset(start_from_random_day)


    def step(self, action):
        assert self.action_space.contains(action), "Invalid Action"

        TimeManager().step()

        self.heating_system.step(action)

        self.temperature_manager.step(self.heating_system)

        self.occupacy_manager.step()

        # TODO: Calculate reward for people being home or not + hvac usage penalty
        target_temperature = ConfigurationManager().get_temp_config("target_temperature")
        current_temperature = self.temperature_manager.get_current_temperature()
        reward = self.calculate_reward(current_temperature, target_temperature)

        self.total_reward += reward

        done = TimeManager().is_over()
        truncated = False
        info = {}

        # To comply with the gym API, the step function should return the following: observation, reward, terminated, info
        return self.observation(), reward, done, truncated, info

    def calculate_reward(self, current_temperature, target_temperature):
        comfort_zone = 0.5  # degrees

        # Calculate the absolute difference from the target temperature
        temperature_diff = abs(current_temperature - target_temperature)

        # If within the comfort zone - provide a positive reward
        if temperature_diff <= comfort_zone:
            reward = 1 - (temperature_diff / comfort_zone)# Scales linearly within the comfort zone
        else:
            if current_temperature > target_temperature:
                # Do not penalize if the heating system is off
                penalty = 1 - self.heating_system.get_heat_energy()
            else:
                penalty = -np.exp(temperature_diff - comfort_zone)

            # Normalize penalty to a range or adjust scale as needed for your environment
            reward = max(penalty, -10)  # Example: caps the penalty to -10 for extreme temperature differences

        self.total_reward += reward
        self.reward_data.append(self.total_reward)
        return reward

    def reset(self, start_from_random_day=True):
        self.out_tmp_reader = CSVLineReader(ConfigurationManager().get_settings_config("temperature_data_file"), start_from_random=start_from_random_day)
        starting_time, _ = self.out_tmp_reader.get_next_line()
        starting_time = datetime.strptime(starting_time, '%Y%m%dT%H%M')

        self.target_temperature = ConfigurationManager().get_temp_config("target_temperature")
        self.total_reward = 0
        self.reward_data = [0]

        TimeManager().reset_time_to(starting_time, self.run_for_days)
        self.occupacy_manager = OccupancyManager()
        self.temperature_manager = TemperatureManager(self.out_tmp_reader)
        #TODO: Read from configuration
        self.heating_system = HeatingSystem(H_acc=0.50, H_cool=0.25, H_max=5, H_efficiency=0.8, T_base=3, T_max=27)
        return self.observation(), {}

    def observation(self):
        cur_temp = self.temperature_manager.get_current_temperature()
        out_temp = self.temperature_manager.get_current_outside_temperature()
        occ = self.occupacy_manager.is_occupied()
        heat_energy = self.heating_system.get_heat_energy()
        #TODO: Add occupancy
        return np.array([cur_temp, out_temp, heat_energy]).astype(np.float32)

    def render(self, mode='web'):
        if mode == 'web':
            # Check if the renderer already exists and the server is running
            if hasattr(self, 'renderer') and self.renderer.app.server.running:
                print("Server is already running.")
                return

            self.renderer = SimulationRenderer(self)
            self.renderer.run_server()
        elif mode == 'console':
            print(f"{self.current_time} : {self.temperature_manager.get_current_temperature} *C")

    def simulate(self):
        self.renderer = SimulationRenderer(self)
        self.renderer.run_server()
        self.renderer.run_random_simulation()

    def close(self):
        pass

    def get_temperature_data(self):
        time = TimeManager().get_time_history()
        indoor_temp = self.temperature_manager.get_temperature_history()
        outside_temp = self.temperature_manager.get_outside_temperature_history()
        return time, indoor_temp, outside_temp

    def get_heating_data(self):
        return TimeManager().get_time_history(), self.heating_system.get_heat_history()

    def get_occupancy_data(self):
        return TimeManager().get_time_history(), self.occupacy_manager.get_occupancy_history()

    def get_reward_data(self):
        return TimeManager().get_time_history(), self.reward_data


if __name__ == '__main__':
    env = SmartHomeTempControlEnv()
    env.simulate()
