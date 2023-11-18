CONFIG = {
    'starting_temperature': 22, # degrees Celsius
    'outside_temperature': 10,  # degrees Celsius
    'user_preference': 22,      # degrees Celsius
    'insulation_quality': 0.5,  # coefficient ranging from 0 (no insulation) to 1 (perfect insulation)
    'hvac_efficiency': 1,     # maximum temperature change per time step
    'time_factor': 1/60,         # dilutes the outside temperature's influence to represent a set-length simulation
    'heat_delay': 0.1,          # how fast the heating meter fills up
}
