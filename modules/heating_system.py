import math

class HeatingSystem:
    def __init__(self, H_acc, H_cool, H_max, H_efficiency, T_base, T_max):
        self.H_energy = 0                   # Initial heat energy in the HS
        self.H_acc = H_acc                  # Heat accumulation rate
        self.H_cool = H_cool                # Cooling rate
        self.H_max = H_max                  # Maximum heat energy
        self.H_efficiency = H_efficiency    # Efficiency of heat transfer
        self.T_base = T_base                # Base temperature for exponential calculation:
                                            #   shape of exponential curve (the lesser the value, the steeper the curve)
        self.T_max = T_max                  # Maximum comfortable temperature

    def update_heat_energy(self, heating_on):
        """Update the HS's heat energy based on its current state."""
        if heating_on:
            self.H_energy = min(self.H_energy + self.H_acc, self.H_max)
        else:
            self.H_energy = max(self.H_energy - self.H_cool, 0)

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


# Example usage
if __name__ == "__main__":
    HS = HeatingSystem(H_acc=0.1, H_cool=0.05, H_max=5, H_efficiency=0.8, T_base=3, T_max=27)

    T_current = 20              # Starting room temperature in degrees Celsius
    minutes = 60 * 24           # Simulate for 24 hours

    for minute in range(minutes):
        # Simulate turning the heating on for the first 12 hours
        if minute < 60 * 12:
            HS.update_heat_energy(heating_on=True)
        else:
            # And then off for the next 12 hours
            HS.update_heat_energy(heating_on=False)

        T_change = HS.get_temperature_change(T_current)
        T_current += T_change / 60

        if minute % 60 == 0:
            print(f"Minute {minute // 60 }: Temperature = {T_current:.2f}°C, Heat energy = {HS.H_energy:.2f}, temperature change = {T_change:.4f}°C")
