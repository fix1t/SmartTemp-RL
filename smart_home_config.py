CONFIG = {
    'starting_temperature': 20, # degrees Celsius
    'outside_temperature': 10,  # degrees Celsius
    'user_preference': 22,      # degrees Celsius
    'insulation_quality': 0.1,  # coefficient ranging from 0 (perfect insulation) to 1 (no insulation)
    'heater_at_max': 40,        # maximum temperature output of the heater
    'cooler_at_max': 5,        # minimal temperature output of the cooler
    'hvac_efficiency': 0.05,     # how much of the heater/cooler's output is actually used
    'time_factor': 1/60,        # dilutes the outside temperature's influence to represent a step-length simulation
    'meter_step': 0.25,          # how fast the meter fills up

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
    'random_event_chance': 0.1, # 5% chance someone stays home or has an irregular schedule
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
