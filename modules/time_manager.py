from datetime import timedelta, datetime

from modules.configuration_manager import ConfigurationManager

class TimeManager:
    """
    Singleton class to manage simulation time
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TimeManager, cls).__new__(cls)
            cls._instance.load_configuration()
        return cls._instance

    def load_configuration(self):
        self.current_time = ConfigurationManager().get_settings_config("start_of_simulation")
        self.today = self.current_time.strftime("%A")
        self.time_history = []
        self.time_history.append(self.current_time)

    def get_today(self):
        return self.today

    def get_current_time(self):
        return self.current_time

    def update_time(self):
        self.current_time += timedelta(minutes=ConfigurationManager().get_settings_config("minutes_per_step"))
        self.time_history.append(self.current_time)
        return self.current_time

    def get_time_history(self):
        return self.time_history

    def step(self):
        self.update_time()
        return self.current_time
