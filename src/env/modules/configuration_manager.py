
"""
Singleton class to manage configuration settings
"""
import json
class ConfigurationManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ConfigurationManager, cls).__new__(cls)
            cls._instance.load_configuration()
        return cls._instance

    def load_configuration(self, path='env/environment_configuration.json'):
        with open(path, 'r') as file:
            self.config = json.load(file)
        return self.config

    def get_temp_config(self, key):
        return self.config.get('temperature').get(key)

    def get_settings_config(self, key):
        return self.config.get('settings').get(key)

    def get_schedule_config(self, key):
        return self.config.get('schedule').get(key)

    def get_schedule_for_person_on_day(self, person, day):
        return self.config.get('schedule').get('weekly_schedule').get(day).get(person)


    def set_config(self, value, *keys):
        config = self.config
        # Traverse through the keys, going deeper into the dictionary
        # until we reach the last key.
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}  # Create a new dictionary if the key does not exist
            config = config[key]

        # The last key in the sequence is where to set the value.
        last_key = keys[-1]
        config[last_key] = value
