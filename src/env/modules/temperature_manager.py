from env.tools.csv_line_reader import CSVLineReader
from env.tools.logger_manager import Logger
from env.modules.configuration_manager import ConfigurationManager
from env.modules.time_manager import TimeManager

class TemperatureManager:
    """
    Holds the current temperature and the outside temperature
    Updates the temperature based on the outside temperature and the HVAC system
    Updates the outside temperature over time
    """
    def __init__(self, temperature_data_file_reader: CSVLineReader):
        self.cur_temp = ConfigurationManager().get_temp_config("starting_temperature")       # inside temperature


        self.out_temp_reader = temperature_data_file_reader

        self.out_temp = self.out_temp_reader.get_next_line()[1]                     # outside temperature
        self.out_temp_next = self.out_temp_reader.get_next_line()[1]
        self.out_temp_step_diff = self.out_temp_next - self.out_temp

        self.step_counter = 0

        self.temp_history = []                                                          # inside temp history
        self.out_temp_history = []                                                      # outside temperature history

    def out_temp_update(self):
        """
        The outside temperature linearly changes in between the records (hourly data)
        Updates outside temperature, next temperature, diff and the step counter
        """
        record_step = ConfigurationManager().get_settings_config("outside_temperature_record_step")

        if self.step_counter >= record_step:

            # get next outside temperature
            self.out_temp_next = self.out_temp_reader.get_next_line()[1]
            if self.out_temp_next is None:
                self.out_temp_reader.reset_to_beginning()
                self.out_temp_next = self.out_temp_reader.get_next_line()[1]
                Logger().warning("Out of data. Resseting to the beginning of the file.")

            MINUTES_PER_STEP = ConfigurationManager().get_settings_config("minutes_per_step")
            HOUR_MINUTES = 60
            TIME_FACTOR = MINUTES_PER_STEP / HOUR_MINUTES
            self.out_temp_step_diff = (self.out_temp_next - self.out_temp) * TIME_FACTOR
            self.step_counter -= record_step

        self.out_temp += self.out_temp_step_diff
        self.step_counter += ConfigurationManager().get_settings_config("minutes_per_step")

    def update_temp(self, heating_system):
        """
        Updates the temperature based on the outside temperature and the HVAC system
        """
        INSULATION = ConfigurationManager().get_temp_config("insulation_quality")
        MINUTES_PER_STEP = ConfigurationManager().get_settings_config("minutes_per_step")
        HOUR_MINUTES = 60
        TIME_FACTOR = MINUTES_PER_STEP / HOUR_MINUTES

        # Change due to outside temperature
        outside_temp_change = (self.out_temp - self.cur_temp) * (1 - INSULATION) * TIME_FACTOR
        # Change due to HVAC system
        #TODO: heating only for now
        hvac_temp_change = heating_system.get_temperature_change(self.cur_temp)
        new_temerature = self.cur_temp + (outside_temp_change + hvac_temp_change)
        Logger().info(f"{TimeManager().get_current_time()} : New temp {new_temerature} = current {self.cur_temp} + outside {outside_temp_change} + hvac {hvac_temp_change}")
        self.cur_temp = new_temerature

    def append_temp_history(self):
        self.temp_history.append(self.cur_temp)
        self.out_temp_history.append(self.out_temp)

    def step(self, heating_system):
        """
        Abstraction of the temperature step in the simulation
        Updates the temperature, the outside temperature and the temperature history
        """
        self.update_temp(heating_system)
        self.out_temp_update()
        self.append_temp_history()

    def get_temperature_history(self):
        return self.temp_history

    def get_outside_temperature_history(self):
        return self.out_temp_history

    def get_current_temperature(self):
        return self.cur_temp

    def get_current_outside_temperature(self):
        return self.out_temp
