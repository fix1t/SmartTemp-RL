
"""
Singleton class to manage configuration settings
"""
class ConfigurationManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ConfigurationManager, cls).__new__(cls)
            cls._instance.load_configuration()
        return cls._instance

    def load_configuration(self):
        from smart_home_config import CONFIG
        self.config = CONFIG

    def get_temp_config(self, key):
        return self.config.get('temperature').get(key)

    def get_settings_config(self, key):
        return self.config.get('settings').get(key)

    def get_schedule_config(self, key):
        return self.config.get('schedule').get(key)

    def get_schedule_for_person_on_day(self, person, day):
        return self.config.get('schedule').get('weekly_schedule').get(day).get(person)

    def set_config(self, key, value):
        self.config[key] = value
