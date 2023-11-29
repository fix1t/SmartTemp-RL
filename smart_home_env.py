import gym
from gym import spaces
import numpy as np
from datetime import datetime, timedelta
import random

from smart_home_config import CONFIG

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

        # Define weekly schedule
        self.schedule = CONFIG['weekly_schedule']
        self.random_event = CONFIG['random_event']
        self.random_event_chance = CONFIG['random_event_chance']
        self.random_event_max_duration = CONFIG['random_event_max_duration']
        self.people_presence = {
            "father": True,
            "mother": True,
            "child": True
        }

        # Initial time 1/1/2020 00:00
        self.current_time = datetime(2020, 1, 1, 0, 0)
        self.current_day = self.current_time.strftime('%A')

        # Heating and Cooling meters
        self.heating_meter = 0.0
        self.cooling_meter = 0.0
        self.max_meter = 10.0

    def step(self, action):
        assert self.action_space.contains(action), "Invalid Action"
        self.update_time()

        self.execute_action(action)

        self. update_temperature()

        # Update people presence
        self.update_people_presence()

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

    def update_people_presence(self):
        for person, schedule in CONFIG['people'].items():
            if random.random() < CONFIG['random_home_day_chance']:
                self.people_presence[person] = random.choice([True, False])
                continue

            leave_time = self.parse_time(schedule['leave'])
            return_time = self.parse_time(schedule['return']) if isinstance(schedule['return'], str) else self.parse_weekday_return_time(schedule['return'])

            # Add randomness to the schedule
            leave_time += timedelta(minutes=random.randint(-schedule['variance'], schedule['variance']))
            return_time += timedelta(minutes=random.randint(-schedule['variance'], schedule['variance']))

            self.people_presence[person] = not (leave_time <= self.current_time < return_time)

    def update_time(self):
        self.current_time += timedelta(minutes=1)
        today = self.current_time.strftime('%A')
        if self.current_day != today:
            self.current_day = today
            self.set_schedule()

    def set_schedule(self):
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

    def update_temperature(self):
        print(f"Current temp: {self.current_temperature}")
        outside_temp_change = (self.outside_temperature - self.current_temperature) * self.insulation_factor * self.time_factor
        print(f"Outside temp change: {outside_temp_change}")
        hvac_temp_change = (self.heating_meter/10 * self.heater_at_max - self.cooling_meter/10 * self.cooler_at_max) * self.hvac_efficiency * self.time_factor
        print(f"HVAC temp change: {hvac_temp_change}")
        total_temp_change = outside_temp_change + hvac_temp_change
        print(f"Total temp change: {total_temp_change}")
        self.current_temperature += total_temp_change

    def parse_time(self, time_str):
        hour, minute = map(int, time_str.split(':'))
        return self.current_time.replace(hour=hour, minute=minute)

    def parse_weekday_return_time(self, return_times):
        weekday = self.current_time.strftime('%a')
        return self.parse_time(return_times.get(weekday, '15:00'))  # default return time if not specified
