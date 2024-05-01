import math
import time
import gym
from gym import spaces
import numpy as np
from datetime import datetime

from env.modules.occupancy_manager import OccupancyManager
from env.modules.temperature_manager import TemperatureManager
from env.modules.configuration_manager import ConfigurationManager
from env.modules.time_manager import TimeManager
from env.modules.heating_system import HeatingSystem
from env.tools.renderer import SimulationRenderer
from env.tools.csv_line_reader import CSVLineReader

class TempRegulationEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, start_from_random_day=True, max_steps_per_episode=7 * 24 * 4, seed=None,
                 config_file='env/environment_configuration.json', temp_data_file='data/basel_10_years_hourly.csv'):
        super(TempRegulationEnv, self).__init__()
        self.action_space = spaces.Discrete(5)

        if seed is not None:
            np.random.seed(seed)

        indoor_temp, outoor_temp, occupancy, heat_energy, hour, weekday = {}, {}, {}, {}, {}, {}
        indoor_temp['min'], indoor_temp['max'] = 0, 30
        outoor_temp['min'], outoor_temp['max'] = -30, 40
        occupancy['min'], occupancy['max'] = 0, 1
        heat_energy['min'], heat_energy['max'] = 0, 5
        hour['min'], hour['max'] = 0, 24
        weekday['min'], weekday['max'] = 0, 6

        low = np.array([indoor_temp['min'], outoor_temp['min'], occupancy['min'], heat_energy['min'], hour['min'], weekday['min']])
        high = np.array([indoor_temp['max'], outoor_temp['max'], occupancy['max'], heat_energy['max'], hour['max'], weekday['max']])

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.total_reward = 0
        self.max_steps_per_episode = max_steps_per_episode
        self.start_from_random_day = start_from_random_day

        ConfigurationManager().load_configuration(config_file)
        ConfigurationManager().set_config(temp_data_file, "settings", "temperature_data_file")

        self.reset(start_from_random_day)


    def step(self, action):
        assert self.action_space.contains(action), "Invalid Action"

        TimeManager().step()

        self.heating_system.step(action)

        self.temperature_manager.step(self.heating_system)

        self.occupacy_manager.step()

        reward = self.calculate_reward()

        self.total_reward += reward
        self.reward_data['total_reward'].append(self.total_reward)

        done = TimeManager().is_over()
        truncated = False
        info = {}

        # To comply with the gym API, the step function should return the following: observation, reward, terminated, info
        return self.observation(), reward, done, truncated, info

    def calculate_reward(self):
        is_occupied = self.occupacy_manager.is_occupied()


        # If the house is not occupied, we do not penalize the temperature difference
        if not is_occupied:
            energy_reward = self.energy_reward() # Penalize more when the house is not occupied
            temperature_reward = 0
            reward = energy_reward
        else:
            COMFORT_TO_COST_PREFERENCE = 0.65
            energy_reward =  (1- COMFORT_TO_COST_PREFERENCE) * self.energy_reward()
            temperature_reward = COMFORT_TO_COST_PREFERENCE * self.temperature_reward()
            reward = energy_reward + temperature_reward

        self.reward_data['energy_reward'].append(energy_reward)
        self.reward_data['temperature_reward'].append(temperature_reward)
        self.reward_data['step_reward'].append(reward)
        return reward

    def temperature_reward(self):
        target_temperature = ConfigurationManager().get_temp_config("target_temperature")
        current_temperature = self.temperature_manager.get_current_temperature()

        # Calculate the absolute difference from the target temperature
        temperature_diff = abs(current_temperature - target_temperature)

        # Give extra motivation, when someone arrives - to heat up the house in advance
        if self.occupacy_manager.is_arrival():
            EXTRA_MOTIVATION = 5
        else:
            EXTRA_MOTIVATION = 1

        MAXIMUM_TEMP_REWARD = 10  # Example value
        MINIMUM_TEMP_REWARD = -10  # Example value
        COMFORT_ZONE = 1  # Example value

        # < -10; 0; 10 >
        # Hyperbolic function, where reward is 0 at comfort zone
        reward = (MAXIMUM_TEMP_REWARD * COMFORT_ZONE/ temperature_diff) + MINIMUM_TEMP_REWARD
        reward = min(MAXIMUM_TEMP_REWARD, reward)

        return EXTRA_MOTIVATION * reward


    def energy_reward(self):
        normalized_energy = self.heating_system.get_heat_energy()/self.heating_system.H_max
        # Rewards < -10; 0 >
        return -1 * normalized_energy * 10

    def reset(self, start_from_random_day=None):
        if start_from_random_day is None:
            start_from_random_day = self.start_from_random_day
        self.out_tmp_reader = CSVLineReader(
            ConfigurationManager().get_settings_config("temperature_data_file"),
            start_from_random=start_from_random_day,
            seed=np.random.randint(0, 10_000)
            )

        starting_time, _ = self.out_tmp_reader.get_next_line()
        starting_time = datetime.strptime(starting_time, '%Y%m%dT%H%M')

        self.target_temperature = ConfigurationManager().get_temp_config("target_temperature")
        self.total_reward = 0
        self.reward_data = {}
        self.reward_data['step_reward'] = []
        self.reward_data['energy_reward'] = []
        self.reward_data['temperature_reward'] = []
        self.reward_data['total_reward'] = []

        TimeManager().reset_time_to(starting_time, self.max_steps_per_episode)
        self.occupacy_manager = OccupancyManager()
        self.temperature_manager = TemperatureManager(self.out_tmp_reader)
        #TODO: Read from configuration
        self.heating_system = HeatingSystem(H_acc_base=0.5, H_cool=1, H_max=5, H_efficiency=0.8, T_base=3, T_max=27)
        return self.observation(), {}

    def set_max_steps_per_episode(self, max_steps_per_episode):
        self.max_steps_per_episode = max_steps_per_episode

    def observation(self):
        cur_temp = self.temperature_manager.get_current_temperature()
        out_temp = self.temperature_manager.get_current_outside_temperature()
        occ = self.occupacy_manager.is_occupied()
        heat_energy = self.heating_system.get_heat_energy()
        hour = TimeManager().get_current_hour()
        weekday = TimeManager().get_weekday()
        return np.array([cur_temp, out_temp, occ, heat_energy, hour, weekday]).astype(np.float32)

    def render(self, mode='web'):
        if mode == 'web':
            # Check if the renderer already exists and the server is running
            if hasattr(self, 'renderer'):
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
    env = TempRegulationEnv()
    env.simulate()
