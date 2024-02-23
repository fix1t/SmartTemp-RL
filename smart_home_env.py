import gym
from gym import spaces
import numpy as np

from modules.occupancy_manager import OccupancyManager
from modules.temperature_manager import TemperatureManager
from modules.configuration_manager import ConfigurationManager
from modules.time_manager import TimeManager
from modules.heating_system import HeatingSystem
from tools.renderer import SimulationRenderer

class SmartHomeTempControlEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self):
        super(SmartHomeTempControlEnv, self).__init__()
        self.action_space = spaces.Discrete(2)  # 0: Heat Up, 1: Do Nothing

        #TODO: Define observation space - Current temperature, outside temperature, occupancy, heating system energy
        self.observation_space = spaces.Box(low=np.array([0, -30,0]), high=np.array([30, 40, 5]), dtype=np.float32)

        self.timeManager = TimeManager()
        self.occupacy_manager = OccupancyManager()
        self.temperature_manager = TemperatureManager()
        self.heating_system = HeatingSystem(H_acc=0.25, H_cool=0.05, H_max=5, H_efficiency=0.8, T_base=3, T_max=27)


    def step(self, action):
        assert self.action_space.contains(action), "Invalid Action"

        self.timeManager.step()

        self.heating_system.step(action)

        self.temperature_manager.step(self.heating_system)

        self.occupacy_manager.step()

        # TODO: Calculate reward for people being home or not + hvac usage penalty
        target_temperature = ConfigurationManager().get_temp_config("target_temperature")
        current_temperature = self.temperature_manager.get_current_temperature()
        reward = self.calculate_reward(current_temperature, target_temperature)

        done = False
        info = {}

        return self.observation(), reward, done, info

    def calculate_reward(self, current_temperature, target_temperature):
        # Define a temperature threshold within which we consider the temperature to be 'comfortable'
        comfort_zone = 0.5  # degrees

        # Calculate the absolute difference from the target temperature
        temperature_diff = abs(current_temperature - target_temperature)

        # If within the comfort zone, provide a positive reward
        if temperature_diff <= comfort_zone:
            return 1 - (temperature_diff / comfort_zone)  # Scales linearly within the comfort zone

        # Outside the comfort zone, use a negative exponential penalty to provide a smooth gradient
        else:
            penalty = -np.exp(temperature_diff - comfort_zone)

            # Normalize penalty to a range or adjust scale as needed for your environment
            normalized_penalty = max(penalty, -10)  # Example: caps the penalty to -10 for extreme temperature differences

            return normalized_penalty


        # if self.occupacy_manager.is_occupied():
        #     if current_temperature < target_temperature + 0.5 and current_temperature > target_temperature - 0.5:
        #         return 5
        #     else:
        #         return -abs(self.temperature_manager.get_current_temperature() - ConfigurationManager().get_temp_config("target_temperature"))
        # else:
        #     return self.heating_system.get_heat_energy() * -1

    def reset(self):
        self.current_time = ConfigurationManager().get_settings_config("start_of_simulation")
        self.occupacy_manager = OccupancyManager()
        self.temperature_manager = TemperatureManager()
        return self.observation()

    def observation(self):
        cur_temp = self.temperature_manager.get_current_temperature()
        out_temp = self.temperature_manager.get_current_outside_temperature()
        occ = self.occupacy_manager.is_occupied()
        heat_energy = self.heating_system.get_heat_energy()
        #TODO: Add occupancy
        return np.array([cur_temp, out_temp, heat_energy]).astype(np.float32)

    # def render(self, mode='console'):
    #     if mode == 'console':
    #         print(f"{self.current_time} : {self.temperature_manager.get_current_temperature} *C")

    def render(self, mode='web'):
        if mode == 'web':
            # Check if the renderer already exists and the server is running
            if hasattr(self, 'renderer') and self.renderer.app.server.running:
                print("Server is already running.")
                return

            self.renderer = SimulationRenderer(self)
            self.renderer.run_server()

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


if __name__ == '__main__':
    env = SmartHomeTempControlEnv()
    env.render('web')


