"""
    File: heating_system.py
    Author: Gabriel Biel

    Description: Module to simulate a heating system that accumulates heat energy
    and transfers it to the environment to regulate the temperature. The system
    has a maximum heat energy capacity and efficiency of heat transfer, and it
    cools off when not actively heating. Behaves like radiator unit.
"""

import math

class HeatingSystem:
    def __init__(self, H_acc_base, H_cool, H_max, H_efficiency, T_base, T_max):
        self.H_energy = 0                   # Initial heat energy in the HS
        self.H_acc_base = H_acc_base        # Heat accumulation rate
        self.H_cool = H_cool                # Cooling rate
        self.H_max = H_max                  # Maximum heat energy
        self.H_efficiency = H_efficiency    # Efficiency of heat transfer
        self.T_base = T_base                # Base temperature for exponential calculation:
                                            #   shape of exponential curve (the lesser the value, the steeper the curve)
        self.T_max = T_max                  # Maximum comfortable temperature
        self.H_history = [self.H_energy]    # History of heat energy

    def update_heat_energy(self, action):
        # Adjust the heat energy based on the action
        if action in [1, 2, 3]:  # Heating actions
            self.H_energy += self.H_acc_base * action
            self.H_energy = min(self.H_energy, self.H_max)  # Cap at H_max
        elif action == 4:  # Cooling off
            self.H_energy = max(self.H_energy - self.H_cool, 0)
        # 0 action means maintain, so no change to H_energy.H_cool, 0)

    def get_temperature_change(self, T_current):
        """
        Calculate the exponential change in temperature based on the current heat energy,
        adjusting for diminishing returns as the temperature approaches T_max.
        """
        # Calculate the factor to adjust for diminishing returns
        diminishing_factor = 1 - (T_current / self.T_max)
        diminishing_factor = max(diminishing_factor, 0)  # Ensure non-negative

        # Adjust the temperature change calculation
        return self.H_efficiency * (math.exp(self.H_energy / self.T_base) - 1) * diminishing_factor

    def get_heat_energy(self):
        return self.H_energy

    def get_heat_history(self):
        return self.H_history

    def is_heating(self):
        return self.H_energy > 0.5

    def step(self, action):
        self.update_heat_energy(action)
        self.H_history.append(self.H_energy)

# Example usage
if __name__ == "__main__":
    # Simulate the system
    HS = HeatingSystem(H_acc_base=0.5, H_cool=1, H_max=5, H_efficiency=0.8, T_base=3, T_max=27)
    T_current = 20  # Starting temperature

    for timestep in range(1, 25):  # Simulate for 24 hours
        if timestep <= 8:  # Increase heating in the morning
            action = 2
        elif 8 < timestep <= 14:  # Lower heating during the day
            action = 4
        elif 14 < timestep <= 20:  # Increase heating in the evening
            action = 1
        else:  # Let it cool off in the evening
            action = 4
        HS.step(action)
        T_change = HS.get_temperature_change(T_current)
        T_current += T_change
        print(f"Time {timestep}: Temp = {T_current:.2f}°C, Heat energy = {HS.H_energy:.2f}, Change = {T_change:.4f}°C")
