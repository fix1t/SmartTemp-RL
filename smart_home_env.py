import gym
from gym import spaces
import numpy as np

from modules.occupancy_manager import OccupancyManager
from modules.temperature_manager import TemperatureManager
from modules.configuration_manager import ConfigurationManager
from modules.time_manager import TimeManager
from modules.heating_system import HeatingSystem

class SmartHomeTempControlEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self):
        super(SmartHomeTempControlEnv, self).__init__()
        self.action_space = spaces.Discrete(3)  # 0: Heat Up, 1: Do Nothing

        #TODO: Define observation space - Current temperature and outside temperature (for now)!!
        self.observation_space = spaces.Box(low=np.array([0]), high=np.array([50]), dtype=np.float32)

        self.timeManager = TimeManager()
        self.occupacy_manager = OccupancyManager()
        self.temperature_manager = TemperatureManager()
        self.heating_system = HeatingSystem(H_acc=0.1, H_cool=0.05, H_max=5, H_efficiency=0.8, T_base=3, T_max=27)


    def step(self, action):
        assert self.action_space.contains(action), "Invalid Action"

        self.timeManager.step()

        self.heating_system.step(action)

        self.temperature_manager.step(self.heating_system)

        self.occupacy_manager.step()

        # TODO: Calculate reward for people being home or not + hvac usage penalty
        reward = -abs(self.temperature_manager.get_current_temperature() - ConfigurationManager().get_temp_config("target_temperature"))

        done = False
        info = {}

        return np.array([self.temperature_manager.get_current_temperature()]).astype(np.float32), reward, done, info

    def reset(self):
        self.current_time = ConfigurationManager().get_settings_config("start_of_simulation")
        self.occupacy_manager = OccupancyManager()
        self.temperature_manager = TemperatureManager()

    def render(self, mode='console'):
        if mode == 'console':
            print(f"{self.current_time} : {self.temperature_manager.get_current_temperature} *C")

    def close(self):
        pass

    def get_temperature_data(self):
        time = self.timeManager.get_time_history()
        indoor_temp = self.temperature_manager.get_temperature_history()
        outside_temp = self.temperature_manager.get_outside_temperature_history()
        return time, indoor_temp, outside_temp

    def get_heating_data(self):
        return self.timeManager.get_time_history(), self.heating_system.get_heat_history()

    def get_occupancy_data(self):
        return self.timeManager.get_time_history(), self.occupacy_manager.get_occupancy_history()

