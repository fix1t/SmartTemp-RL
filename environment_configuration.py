from datetime import datetime

CONFIG = {
    "settings": {
        'minutes_per_step': 15,                     # how long a step is in minutes
        'outside_temperature_record_step': 60,      # how often the outside temperature is recorded in minutes
        'start_of_simulation': datetime(2013, 1, 1, 0, 0),# starting time of the simulation
        'temperature_data_file': 'data/basel_10_years_hourly.csv',  # file with the temperature data
    },

    "temperature": {
        'starting_temperature': 20,                 # degrees Celsius
        'outside_temperature': 10,                  # degrees Celsius
        'target_temperature': 22,                   # degrees Celsius
        'insulation_quality': 0.95,                 # coefficient ranging from 0 (no insulation) to 1 (perfect insulation)
        'heater_at_max': 40,                        # maximum temperature output of the heater
        'heating_meter_at_max': 5,                 # maximum heating meter reading
        'cooler_at_max': 5,                         # minimal temperature output of the cooler
        'hvac_efficiency': 0.05,                    # how much of the heater/cooler's output is actually used
        'meter_step': 0.25,                         # how fast the meter fills up
    },

    "schedule": {
        "people": ["father", "mother", "child"],    # List of people in the house
        "weekly_schedule": {
            "Monday": {
                "father": {"leave": "08:00", "return": "18:00", "variance": 60},
                "mother": {"leave": "07:00", "return": "16:00", "variance": 60},
                "child": {"leave": "07:00", "return": "13:30", "variance": 30}
            },
            "Tuesday": {
                "father": {"leave": "08:00", "return": "18:00", "variance": 60},
                "mother": {"leave": "07:00", "return": "16:00", "variance": 60},
                "child": {"leave": "07:00", "return": "13:30", "variance": 30}
            },
            "Wednesday": {
                "father": {"leave": "08:00", "return": "18:00", "variance": 60},
                "mother": {"leave": "07:00", "return": "16:00", "variance": 60},
                "child": {"leave": "07:00", "return": "13:30", "variance": 30}
            },
            "Thursday": {
                "father": {"leave": "08:00", "return": "18:00", "variance": 60},
                "mother": {"leave": "07:00", "return": "16:00", "variance": 60},
                "child": {"leave": "07:00", "return": "16:00", "variance": 30}
            },
            "Friday": {
                "father": {"leave": "08:00", "return": "18:00", "variance": 60},
                "mother": {"leave": "07:00", "return": "16:00", "variance": 60},
                "child": {"leave": "07:00", "return": "12:30", "variance": 30}
            },
            "Saturday": {
                "father": {"at_home": "true"},
                "mother": {"at_home": "true"},
                "child": {"at_home": "true"}
            },
            "Sunday": {
                "father": {"at_home": "true"},
                "mother": {"at_home": "true"},
                "child": {"at_home": "true"}
            }
        },
        'random_event_chance': 0.1,     # 5% chance someone stays home or has an irregular schedule
        'random_event': {
            'sick-day': 0.05,           # 5% chance someone stays home sick
            'vacation': 0.05,           # 5% chance someone stays home on vacation
            'early-return': 0.2,        # 20% chance someone returns home early
            'late-return': 0.2,         # 20% chance someone returns home late
            'early-leave': 0.2,         # 20% chance someone leaves home early
            'late-leave': 0.2,          # 20% chance someone leaves home late
            'out-of-town': 0.05,        # 5% chance someone is out of town
            'holiday': 0.05,            # 5% chance it's a holiday
        },
        'random_event_max_duration': 180, # Implies only to early/late leave/arrival
    }
}
