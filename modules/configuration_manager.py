
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
        from environment_configuration import CONFIG
        self.config = CONFIG

    def get_temp_config(self, key):
        return self.config.get('temperature').get(key)

    def get_settings_config(self, key):
        return self.config.get('settings').get(key)

    def get_schedule_config(self, key):
        return self.config.get('schedule').get(key)

    def get_schedule_for_person_on_day(self, person, day):
        return self.config.get('schedule').get('weekly_schedule').get(day).get(person)


    def set_config(self, value, key, key2=None, key3=None, key4=None, key5=None):
        if key2:
            if key3:
                if key4:
                    if key5:
                        self.config[key][key2][key3][key4][key5] = value
                    else:
                        self.config[key][key2][key3][key4] = value
                else:
                    self.config[key][key2][key3] = value
            else:
                self.config[key][key2] = value
        else:
            self.config[key] = value
