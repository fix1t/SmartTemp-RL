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

    def __init__(self, start_from_random_day=True, max_steps_per_episode=7 * 24 * 4, seed=None):
        super(TempRegulationEnv, self).__init__()
        self.action_space = spaces.Discrete(2)  # 0: Heat Up, 1: Do Nothing

        if seed is not None:
            np.random.seed(seed)

        #TODO: Define observation space - Current temperature, outside temperature, occupancy, heating system energy
        self.observation_space = spaces.Box(low=np.array([0, -30, 0, 0, 0]), high=np.array([30, 40, 1, 6, 5]), dtype=np.float32)
        self.total_reward = 0
        self.max_steps_per_episode = max_steps_per_episode
        self.start_from_random_day = start_from_random_day
        self.reset(start_from_random_day)


    def step(self, action):
        assert self.action_space.contains(action), "Invalid Action"

        TimeManager().step()

        self.heating_system.step(action)

        self.temperature_manager.step(self.heating_system)

        self.occupacy_manager.step()

        reward = self.calculate_reward()

        self.total_reward += reward
        self.reward_data.append(reward)
        # self.reward_data.append(self.total_reward)

        done = TimeManager().is_over()
        truncated = False
        info = {}

        # To comply with the gym API, the step function should return the following: observation, reward, terminated, info
        return self.observation(), reward, done, truncated, info

    def calculate_reward(self):
        is_occupied = self.occupacy_manager.is_occupied()

        # If the house is not occupied, we do not penalize the temperature difference
        if not is_occupied:
            reward = self.energy_reward() # Penalize more when the house is not occupied
            # print(f"Energy reward: {reward}")
        else:
            COMFORT_TO_COST_PREFERENCE = 0.7
            energy_reward =  (1- COMFORT_TO_COST_PREFERENCE) * self.energy_reward()
            temperature_reward = COMFORT_TO_COST_PREFERENCE * self.temperature_reward()
            reward = energy_reward + temperature_reward
            # print(f"Energy reward: {energy_reward} + Temp reward: {temperature_reward} = {reward}")
        return reward

    def temperature_reward(self):
        target_temperature = ConfigurationManager().get_temp_config("target_temperature")
        current_temperature = self.temperature_manager.get_current_temperature()
        heating_on = self.heating_system.is_heating()

        # Calculate the absolute difference from the target temperature
        temperature_diff = abs(current_temperature - target_temperature)
        MAXIMUM_TEMP_REWARD = 3
        COMFORT_ZONE = 1

        # Give extra motivation, when someone arrives - to heat up the house in advance
        if self.occupacy_manager.is_arrival():
            EXTRA_MOTIVATION = 5
        else:
            EXTRA_MOTIVATION = 1

        #                             | <- High temperature
        #                    OK       |        BAD
        #  Heating off -> ____________|____________ <- Heating on
        #                             |
        #                    BAD      |        OK
        #                             | <- Low temperature

        is_higher_than_target = current_temperature > target_temperature
        is_comfortable = temperature_diff < COMFORT_ZONE

        if not heating_on and is_higher_than_target and not is_comfortable:
            reward = 0
        else:
            reward = MAXIMUM_TEMP_REWARD - temperature_diff / COMFORT_ZONE

        return EXTRA_MOTIVATION * reward


    def energy_reward(self):
        return -self.heating_system.get_heat_energy()

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
        self.reward_data = [0]

        TimeManager().reset_time_to(starting_time, self.max_steps_per_episode)
        self.occupacy_manager = OccupancyManager()
        self.temperature_manager = TemperatureManager(self.out_tmp_reader)
        #TODO: Read from configuration
        self.heating_system = HeatingSystem(H_acc=0.50, H_cool=0.25, H_max=5, H_efficiency=0.8, T_base=3, T_max=27)
        return self.observation(), {}

    def set_max_steps_per_episode(self, max_steps_per_episode):
        self.max_steps_per_episode = max_steps_per_episode

    def observation(self):
        cur_temp = self.temperature_manager.get_current_temperature()
        out_temp = self.temperature_manager.get_current_outside_temperature()
        occ = self.occupacy_manager.is_occupied()
        weekday = TimeManager().get_weekday()
        heat_energy = self.heating_system.get_heat_energy()
        return np.array([cur_temp, out_temp, occ, weekday, heat_energy]).astype(np.float32)

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
