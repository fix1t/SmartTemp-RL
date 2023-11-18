import gym
from gym import spaces
import numpy as np
# from smart_home_config import CONFIG

CONFIG = {
    'starting_temperature': 20, # degrees Celsius
    'outside_temperature': 10,  # degrees Celsius
    'user_preference': 22,      # degrees Celsius
    'insulation_quality': 0.5,  # coefficient ranging from 0 (no insulation) to 1 (perfect insulation)
    'heater_at_max': 80,        # maximum temperature output of the heater
    'cooler_at_max': 15,       # minimal temperature output of the cooler
    'hvac_efficiency': 0.3,     # how much of the heater/cooler's output is actually used
    'time_factor': 1/60,        # dilutes the outside temperature's influence to represent a step-length simulation
    'meter_step': 0.5,          # how fast the meter fills up
}

class SmartHomeTempControlEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self):
        super(SmartHomeTempControlEnv, self).__init__()
        self.action_space = spaces.Discrete(3)  # 0: Heat Up, 1: Cool Down, 2: Do Nothing
        self.observation_space = spaces.Box(low=np.array([0]), high=np.array([50]), dtype=np.float32) # Current temperature
        
        # Define initial conditions
        self.user_preference = CONFIG['user_preference']
        self.current_temperature = CONFIG['starting_temperature']
        self.outside_temperature = CONFIG['outside_temperature']
        self.insulation_factor = CONFIG['insulation_quality']
        self.time_factor = CONFIG['time_factor']
        self.heater_at_max = CONFIG['heater_at_max']
        self.cooler_at_max = CONFIG['cooler_at_max']
        self.hvac_efficiency = CONFIG['hvac_efficiency']
        self.meter_step = CONFIG['meter_step']

        # Heating and Cooling meters
        self.heating_meter = 0.0
        self.cooling_meter = 0.0
        self.max_meter = 10.0

    def step(self, action):
        assert self.action_space.contains(action), "Invalid Action"

        if action == 0:  # Heat up
            self.heating_meter = min(self.max_meter, self.heating_meter + self.meter_step)
            self.cooling_meter = max(0, self.cooling_meter - self.meter_step/2)
        elif action == 1:  # Cool down
            self.heating_meter = max(0, self.heating_meter - self.meter_step/2)
            self.cooling_meter = min(self.max_meter, self.cooling_meter + self.meter_step)
        elif action == 2:  # Do nothing heating
            self.heating_meter = max(0, self.heating_meter - self.meter_step/2)
            self.cooling_meter = max(0, self.cooling_meter - self.meter_step/2)

        # Update current temperature
        # self.current_temperature += (self.outside_temperature - self.current_temperature) * self.insulation_factor * self.time_factor
        # self.current_temperature += (self.heating_meter/10 * self.heater_at_max - self.cooling_meter/10 * self.cooler_at_max) * self.time_factor

        print(f"Current temp: {self.current_temperature}")
        outside_temp_change = (self.outside_temperature - self.current_temperature) * self.insulation_factor * self.time_factor
        print(f"Outside temp change: {outside_temp_change}")
        hvac_temp_change = (self.heating_meter/10 * self.heater_at_max - self.cooling_meter/10 * self.cooler_at_max) * self.hvac_efficiency * self.time_factor
        print(f"HVAC temp change: {hvac_temp_change}")
        total_temp_change = outside_temp_change + hvac_temp_change
        print(f"Total temp change: {total_temp_change}")
        self.current_temperature += total_temp_change


        # Calculate reward
        reward = -abs(self.current_temperature - self.user_preference)
        
        
        done = False
        info = {}

        return np.array([self.current_temperature]).astype(np.float32), reward, done, info

    def reset(self):
        self.current_temperature = CONFIG['starting_temperature']
        self.outside_temperature = CONFIG['outside_temperature']
        self.heating_meter = 0.0
        self.cooling_meter = 0.0
        return np.array([self.current_temperature]).astype(np.float32)

    def render(self, mode='console'):
        if mode == 'console':
            print(f"Current temperature: {self.current_temperature}, Heating Meter: {self.heating_meter}, Cooling Meter: {self.cooling_meter}")

    def close(self):
        pass
