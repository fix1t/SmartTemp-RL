import gym
from gym import spaces
import numpy as np
from datetime import datetime, timedelta
import random

from occupancy_manager import OccupancyManager
from temperature_manager import TemperatureManager
from configuration_manager import ConfigurationManager
from time_manager import TimeManager

class SmartHomeTempControlEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self):
        super(SmartHomeTempControlEnv, self).__init__()
        self.action_space = spaces.Discrete(3)  # 0: Heat Up, 1: Cool Down, 2: Do Nothing

        #TODO: Define observation space - Current temperature and outside temperature (for now)!!
        self.observation_space = spaces.Box(low=np.array([0]), high=np.array([50]), dtype=np.float32)

        self.timeManager = TimeManager()
        self.occupacy_manager = OccupancyManager()
        self.temperature_manager = TemperatureManager()

    def step(self, action):
        assert self.action_space.contains(action), "Invalid Action"

        # self.execute_action(action)

        self.timeManager.time_step()

        self.temperature_manager.temperature_step(1)

        self.occupacy_manager.occupacy_step()

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


    def execute_action(self, action):
        if action == 0:  # Heat up
            self.heating_meter = min(self.max_meter, self.heating_meter + self.meter_step)
            self.cooling_meter = max(0, self.cooling_meter - self.meter_step/2)
        elif action == 1:  # Cool down
            self.heating_meter = max(0, self.heating_meter - self.meter_step/2)
            self.cooling_meter = min(self.max_meter, self.cooling_meter + self.meter_step)
        elif action == 2:  # Do nothing heating
            self.heating_meter = max(0, self.heating_meter - self.meter_step/2)
            self.cooling_meter = max(0, self.cooling_meter - self.meter_step/2)


    def get_temperature_data(self):
        time = self.timeManager.get_time_history()
        indoor_temp = self.temperature_manager.get_temperature_history()
        outside_temp = self.temperature_manager.get_outside_temperature_history()
        return time, indoor_temp, outside_temp

    def get_control_data(self):
        #TODO get from temperature_manager
        return self.timeManager.get_time_history(), self.heating_meter_history, self.cooling_meter_history

    def get_occupancy_data(self):
        return self.timeManager.get_time_history(), self.occupacy_manager.get_occupancy_history()

