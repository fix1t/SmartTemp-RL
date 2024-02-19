from tools.logger_manager import Logger
from modules.configuration_manager import ConfigurationManager
from datetime import datetime, timedelta
import random

from modules.time_manager import TimeManager

class OccupancyManager:
    """
    Manages occupancy of the house
    Updates the occupancy based on the schedule and random events
    Tracks history of occupancy
    """
    def __init__(self):
        self.today = TimeManager().get_today()
        self.people = ConfigurationManager().get_schedule_config('people')

        self.todays_schedule = {}               # schedule for the current day with variance & events applied
        self.generate_schedule()

        self.people_presence_history = {}       # history of people's presence - {person: [True, False, ...]}
        self.people_presence = {}               # current presence of people - {person: True/False}


    def generate_schedule(self):
        """Generate the schedule for the current day - apply variance and random events"""
        for person in self.people:
            Logger().info(f"Generating schedule for {person}")
            print(person)
            self.generate_schedule_for_person(person)



    def generate_schedule_for_person(self, persons_name):
        """Generate the schedule for the current day for a specific person - apply variance and random events"""
        persons_schedule = ConfigurationManager().get_schedule_config('weekly_schedule')[self.today][persons_name]
        Logger().info(f"People's schedule: {persons_schedule} for {persons_name}")

        variance = persons_schedule['variance']
        leave_time = datetime.strptime(persons_schedule['leave'], '%H:%M')
        return_time = datetime.strptime(persons_schedule['return'], '%H:%M')

        leave_time += timedelta(minutes=random.randint(-variance, variance))
        return_time += timedelta(minutes=random.randint(-variance, variance))

        daily_schedule = {
            'leave': leave_time.strftime('%H:%M'),
            'return': return_time.strftime('%H:%M')
        }

        # Check for random events
        if random.random() < ConfigurationManager().get_schedule_config('random_event_chance'):
            random_event_pick = random.random()
            starting_chance = 0

            for event, chance in ConfigurationManager().get_schedule_config('random_event').items():
                if random_event_pick < starting_chance + chance:
                    Logger().info(f"Random event: {event} for {persons_name}")
                    self.apply_random_event(event, daily_schedule)
                    break
                starting_chance += chance

        self.todays_schedule[persons_name] = daily_schedule

    def apply_random_event(self, event, daily_schedule):
        if event == 'sick-day':
            daily_schedule['at_home'] = True
        elif event == 'vacation':
            daily_schedule['at_home'] = True
        elif event == 'early-return':
            return_time = self.parse_time(daily_schedule['return'])
            return_time -= timedelta(minutes=random.randint(0, ConfigurationManager().get_schedule_config('random_event_max_duration')))
            daily_schedule['return'] = return_time.strftime('%H:%M')
        elif event == 'late-return':
            return_time = self.parse_time(daily_schedule['return'])
            return_time += timedelta(minutes=random.randint(0, ConfigurationManager().get_schedule_config('random_event_max_duration')))
            daily_schedule['return'] = return_time.strftime('%H:%M')
        elif event == 'early-leave':
            leave_time = self.parse_time(daily_schedule['leave'])
            leave_time -= timedelta(minutes=random.randint(0, ConfigurationManager().get_schedule_config('random_event_max_duration')))
            daily_schedule['leave'] = leave_time.strftime('%H:%M')
        elif event == 'late-leave':
            leave_time = self.parse_time(daily_schedule['leave'])
            leave_time += timedelta(minutes=random.randint(0, ConfigurationManager().get_schedule_config('random_event_max_duration')))
            daily_schedule['leave'] = leave_time.strftime('%H:%M')
        elif event == 'out-of-town':
            daily_schedule['at_home'] = False
        elif event == 'holiday':
                daily_schedule['at_home'] = False

    def update_people_presence(self):
        for person in self.todays_schedule:
            persons_schedule = self.todays_schedule[person]
            if 'at_home' in persons_schedule:
                if persons_schedule['at_home']:
                    self.people_presence[person] = True
                else:
                    self.people_presence[person] = False
            else:
                leave_time = self.parse_time(persons_schedule['leave'])
                return_time = self.parse_time(persons_schedule['return'])
                if leave_time <= TimeManager().get_current_time() <= return_time:
                    self.people_presence[person] = False
                else:
                    self.people_presence[person] = True

    def parse_time(self, time_str):
        hour, minute = map(int, time_str.split(':'))
        return TimeManager().get_current_time().replace(hour=hour, minute=minute)

    def step(self):
        """Update the occupancy based on the schedule and random events"""
        # Check if the day has changed - if so, generate new schedule
        if self.today != TimeManager().get_today():
            self.today = TimeManager().get_today()
            self.generate_schedule()

        self.update_people_presence()
        self.append_presence_history()

        return self.people_presence

    def append_presence_history(self):
        for person in self.people:
            # Check if the person's presence status is known before attempting to log it
            if person not in self.people_presence:
                Logger().info(f"{TimeManager().get_current_time()} : Presence status of {person} unknown. Initializing.")
                self.people_presence[person] = False  # Or some default value as appropriate

            if person not in self.people_presence_history:
                Logger().info(f"{TimeManager().get_current_time()} : {person} NOT in history. Adding.")
                self.people_presence_history[person] = [self.people_presence[person]]
            else:
                self.people_presence_history[person].append(self.people_presence[person])


    def get_occupancy_history(self):
        return self.people_presence_history


