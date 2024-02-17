from csv_line_reader import CSVLineReader
from configuration_manager import ConfigurationManager
from logger_manager import Logger

"""
Holds the current temperature and the outside temperature
Updates the temperature based on the outside temperature and the HVAC system
Updates the outside temperature over time
"""
class TemperatureManager:
    def __init__(self ):
        self.cur_temp = ConfigurationManager().get_temp_config("starting_temperature")       # inside temperature


        self.out_temp_reader = CSVLineReader("temperature_data/basel_10_years_hourly.csv")

        self.out_temp = self.out_temp_reader.get_next_line()[1]                     # outside temperature
        self.out_temp_next = self.out_temp_reader.get_next_line()[1]
        self.out_temp_step_diff = self.out_temp_next - self.out_temp

        self.step_counter = 0

        self.temp_history = []                                                          # inside temp history
        self.out_temp_history = []                                                      # outside temperature history

    """
    The outside temperature linearly changes in between the records (hourly data)
    Updates outside temperature, next temperature, diff and the step counter
    """
    def out_temp_update(self):
        record_step = ConfigurationManager().get_settings_config("outside_temperature_record_step")

        if self.step_counter >= record_step:

            # get next outside temperature
            self.out_temp_next = self.out_temp_reader.get_next_line()[1]
            if self.out_temp_next is None:
                self.out_temp_reader.reset_to_beginning()
                self.out_temp_next = self.out_temp_reader.get_next_line()[1]
                Logger().warning("Out of data. Resseting to the beginning of the file.")

            self.out_temp_step_diff = (self.out_temp_next - self.out_temp) * self.time_factor
            self.step_counter -= record_step

        self.out_temp += self.out_temp_step_diff
        self.step_counter += ConfigurationManager().get_settings_config("minutes_per_step")

    """
    Updates the temperature based on the outside temperature and the HVAC system
    """
    def update_temp(self, heating_meter):
        INSULATION = ConfigurationManager().get_temp_config("insulation_quality")
        MINUTES_PER_STEP = ConfigurationManager().get_settings_config("minutes_per_step")
        HOUR_MINUTES = 60
        TIME_FACTOR = MINUTES_PER_STEP / HOUR_MINUTES
        HEATER_AT_MAX = ConfigurationManager().get_temp_config("heater_at_max")
        HEATING_METER_AT_MAX = ConfigurationManager().get_temp_config("heating_meter_at_max")
        HVAC_EFFICIENCY = ConfigurationManager().get_temp_config("hvac_efficiency")

        # Change due to outside temperature
        outside_temp_change = (self.out_temp - self.cur_temp) * (1 - INSULATION) * TIME_FACTOR
        # Change due to HVAC system
        #TODO: heating only for now
        hvac_temp_change = (heating_meter/HEATING_METER_AT_MAX * HEATER_AT_MAX) * HVAC_EFFICIENCY * TIME_FACTOR
        new_temerature = self.cur_temp + (outside_temp_change + hvac_temp_change)
        Logger().info(f"TEMPERATURE: {new_temerature} = current {self.cur_temp} + outside {outside_temp_change} + hvac {hvac_temp_change}")
        self.cur_temp = new_temerature
        self.out_temp()

    def append_temp_history(self):
        self.temp_history.append(self.cur_temp)
        self.out_temp_history.append(self.out_temp)

    """
    Abstraction of the temperature step in the simulation
    Updates the temperature, the outside temperature and the temperature history
    """
    def temperature_step(self, heating_meter_status):
        self.update_temp(heating_meter_status)
        self.append_temp_history()
        self.out_temp_update()

    def get_temperature_data(self):
        return self.temp_history, self.out_temp_history
