# SmartTemp-RL

## Run simulation

to install required libraries:
```shell
pip install -r requirements.txt
```

to run simultaion prototype:
```shell
python3 gui.py
```

## Architecture overview
1. **Temperature Update Rule**:
   The temperature in the room approaches the outside temperature over time, influenced by the insulation quality.

2. **Actions**:
   - **Heating/Cooling Meter**: Variable for each that represents the current power level of the heating and cooling system.
   - **Action Dynamics**:
   - Pressing the heating button increases the **heating meter** by a certain amount each minute until a maximum value is reached.
   - Releasing the heating button decreases the heating meter gradually, representing the system cooling down.
   - Similarly, pressing the cooling button increases the **cooling meter**, and releasing it causes the meter to decrease over time.

- **Temperature Adjustment**:
  - The rate of temperature change is proportional to the heating or cooling meter value.
  - This change is applied to the current temperature each minute, considering insulation and outside temperature.

3. **Delay in Action Effects**:
   There's a lag between the action (pressing a lever) and the effect on temperature due to the time it takes for the HVAC system to heat up/cool down.

4. **Step Timing**:
   Each step in the simulation represents **one minute** of real-time.

5. **Goal**:
   Maintain the temperature as close to the user-preferred temperature (`user_preference`) as possible.

6. **Temperature Dynamics**:
   A simple formula for the temperature change:

   ```python
   current_temperature += (outside_temperature - current_temperature) * insulation_factor * time_factor
   ```

   where `insulation_factor` is a measure of how well insulated the room is and `time_factor` dilutes the outside temperature's influence to represent a minute-by-minute simulation.

7. **Heating/Cooling Effects**:
   Apply a predefined amount of temperature change when heating or cooling is turned on, with a delay factor to simulate system response time.

## Daily Schedule and Random Events

### Weekly Schedule
The simulation incorporates a detailed weekly schedule for each family member (father, mother, child), including their daily routines from Monday to Sunday. This schedule includes specific leave and return times on weekdays, with a variance to account for unpredictability in their routine. On weekends, it is assumed they are at home.

```json
{
    "weekly_schedule": {
        "Monday": {
            "father": {"leave": "08:00", "return": "18:00", "variance": 60},
            ... // Rest of the people.
        },
        // Rest of the week.
    },
    "random_home_day_chance": 0.05, // 5% chance someone stays home or has an irregular schedule
    "random_event": {
        "sick-day": 0.05,           // 5% chance someone stays home sick
        "vacation": 0.05,           // 5% chance someone stays home on vacation
        ... // Others..
    },
    "random_event_max_duration": 180 // Implies only to early/late leave/arrival
}
```

### Random Events
The simulation also includes a probability model for random events that can affect the daily routine of the family members. These events include sick days, vacation days, early or late returns, early or late departures, out-of-town trips, and holidays. Each event has a specified probability and a maximum duration, adding an element of unpredictability and realism to the simulation.
