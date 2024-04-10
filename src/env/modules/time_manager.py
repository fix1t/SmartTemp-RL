import datetime as dt

from env.modules.configuration_manager import ConfigurationManager

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
        self.final_time = None
        self.time_history = []
        self.time_history.append(self.current_time)

    def get_today(self):
        return self.current_time.strftime("%A")

    def get_weekday(self):
        return self.current_time.weekday()

    def get_current_hour(self):
        return self.current_time.hour

    def get_current_month(self):
        return self.current_time.month

    def get_current_time(self):
        return self.current_time

    def update_time(self):
        self.current_time += dt.timedelta(minutes=ConfigurationManager().get_settings_config("minutes_per_step"))
        self.time_history.append(self.current_time)
        return self.current_time

    def get_time_history(self):
        return self.time_history

    def step(self):
        self.update_time()
        return self.current_time

    def is_over(self):
        return self.current_time >= self.final_time

    def reset_time_to(self, time, timesteps):
        self.current_time = time
        self.final_time = time + dt.timedelta(minutes=15*timesteps)
        self.time_history = []
        self.time_history.append(self.current_time)
        return self.current_time
