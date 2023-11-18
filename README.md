# SmartTemp-RL

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
