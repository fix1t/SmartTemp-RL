import gym
from gym import spaces
import numpy as np
from datetime import datetime, timedelta
import random

from smart_home_config import CONFIG
from csv_line_reader import CSVLineReader

class SmartHomeTempControlEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, config=CONFIG):
        super(SmartHomeTempControlEnv, self).__init__()
        self.action_space = spaces.Discrete(3)  # 0: Heat Up, 1: Cool Down, 2: Do Nothing

        #TODO: Define observation space - Current temperature and outside temperature (for now)!!
        self.observation_space = spaces.Box(low=np.array([0]), high=np.array([50]), dtype=np.float32)

        # Define initial conditions
        self.user_preference = config['user_preference']
        self.current_temperature = config['starting_temperature']
        self.insulation_factor = config['insulation_quality']
        self.time_factor = config['time_factor']
        self.heater_at_max = config['heater_at_max']
        self.cooler_at_max = config['cooler_at_max']
        self.hvac_efficiency = config['hvac_efficiency']
        self.meter_step = config['meter_step']

        # Define weekly schedule
        self.weekly_schedule = config['weekly_schedule']
        self.random_event = config['random_event']
        self.random_event_chance = config['random_event_chance']
        self.random_event_max_duration = config['random_event_max_duration']
        self.people_presence = {
            "father": False,
            "mother": False,
            "child": False,
        }

        # Initial time 1/1/2020 00:00
        # TODO: Get this from outside temperature source
        self.current_time = datetime(2020, 1, 1, 0, 0)
        self.current_day = self.current_time.strftime('%A')

        # Schedule for the current day with random events and noise applied
        self.schedule = self.generate_schedule()

        # Heating and Cooling meters
        self.heating_meter = 0.0
        self.cooling_meter = 0.0
        self.max_meter = 10.0

        # History for plotting
        self.heating_meter_history = []
        self.cooling_meter_history = []
        self.people_presence_history = {}
        self.time_history = []

    def step(self, action):
        assert self.action_space.contains(action), "Invalid Action"
        self.update_time()

        self.execute_action(action)

        self.update_temperature()

        self.update_people_presence()

        self.append_to_history()

        # TODO: Calculate reward for people being home or not + hvac usage penalty
        reward = -abs(self.current_temperature - self.user_preference)

        done = False
        info = {}


        return np.array([self.current_temperature]).astype(np.float32), reward, done, info

    def reset(self):
        self.outside_temperature_reader.reset_to_beginning()
        self.current_temperature = CONFIG['starting_temperature']
        try:
            self.current_outside_temperature = float(self.outside_temperature_reader.get_next_line()[1])
            self.outside_temperature_next = float(self.outside_temperature_reader.get_next_line()[1])
        except ValueError:
            self.current_outside_temperature = 0
            print("[ERROR]  The string does not represent a valid floating-point number.")
        self.outside_temperature_step_diff = (self.outside_temperature_next - self.current_outside_temperature) * self.time_factor
        self.step_counter = 0

        self.heating_meter = 0.0
        self.cooling_meter = 0.0

        self.current_time = datetime(2020, 1, 1, 0, 0)
        self.current_day = self.current_time.strftime('%A')

        self.schedule = self.generate_schedule()

        self.temperature_history = []
        self.heating_meter_history = []
        self.cooling_meter_history = []
        self.time_history = []
        return np.array([self.current_temperature]).astype(np.float32)

    def render(self, mode='console'):
        if mode == 'console':
            print(f"Current temperature: {self.current_temperature}, Heating Meter: {self.heating_meter}, Cooling Meter: {self.cooling_meter}")

    def close(self):
        pass

    def update_time(self):
        self.current_time += timedelta(minutes=1)
        today = self.current_time.strftime('%A')
        if self.current_day != today:
            self.current_day = today
            self.schedule = self.generate_schedule()



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


    def append_to_history(self):
        self.temperature_history.append(self.current_temperature)
        self.heating_meter_history.append(self.heating_meter)
        self.cooling_meter_history.append(self.cooling_meter)
        self.time_history.append(self.current_time.strftime('%Y-%m-%d %H:%M:%S'))
        self.outside_temperature_history.append(self.current_outside_temperature)
        for person in self.people_presence:
            if person not in self.people_presence_history:
                self.people_presence_history[person] = []
            self.people_presence_history[person].append(self.people_presence[person])

    def get_temperature_data(self):
        return self.time_history, self.temperature_history, self.outside_temperature_history

    def get_control_data(self):
        #TODO get from temperature_manager
        return self.time_history, self.heating_meter_history, self.cooling_meter_history

    def get_occupancy_data(self):
        return self.time_history, self.people_presence_history

