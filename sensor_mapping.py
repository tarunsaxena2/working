"""
sensor_mapping.py — Raw Sensor Unit Conversions
Vaibhav Gautam — Dashboard & Integration
Week 2 Day 1 Task

Converts raw real-world sensor readings to the units
expected by the AI4I dataset and our trained model.

Usage:
    from sensor_mapping import SensorMapper
    mapper = SensorMapper()
    mapped = mapper.map_raw_reading(raw_data)
"""

# =================================================================
# UNIT CONVERSION FUNCTIONS
# =================================================================

def celsius_to_kelvin(celsius: float) -> float:
    """Convert temperature from Celsius to Kelvin."""
    return celsius + 273.15

def kelvin_to_celsius(kelvin: float) -> float:
    """Convert temperature from Kelvin to Celsius."""
    return kelvin - 273.15

def rpm_to_rad_per_sec(rpm: float) -> float:
    """Convert rotational speed from RPM to radians/second."""
    return rpm * (2 * 3.14159 / 60)

def rad_per_sec_to_rpm(rad_per_sec: float) -> float:
    """Convert rotational speed from radians/second to RPM."""
    return rad_per_sec * (60 / (2 * 3.14159))

def nm_to_ft_lbs(nm: float) -> float:
    """Convert torque from Newton-metres to foot-pounds."""
    return nm * 0.737562

def ft_lbs_to_nm(ft_lbs: float) -> float:
    """Convert torque from foot-pounds to Newton-metres."""
    return ft_lbs * 1.35582

def seconds_to_minutes(seconds: float) -> float:
    """Convert tool wear time from seconds to minutes."""
    return seconds / 60

def minutes_to_seconds(minutes: float) -> float:
    """Convert tool wear time from minutes to seconds."""
    return minutes * 60


# =================================================================
# SENSOR MAPPER CLASS
# =================================================================

class SensorMapper:
    """
    Maps raw real-world sensor readings to the exact units
    expected by the AI4I dataset and trained LightGBM model.

    AI4I Dataset expected units:
    - Air temperature:     Kelvin [K]
    - Process temperature: Kelvin [K]
    - Rotational speed:    RPM
    - Torque:              Newton-metres [Nm]
    - Tool wear:           Minutes [min]
    - Type:                String (L/M/H)
    - ambient_temp_C:      Celsius [°C]
    - factory_load_pct:    Percentage [%]
    - humidity_pct:        Percentage [%]
    """

    # Placeholder calibration offsets (to be updated with real sensor data)
    CALIBRATION = {
        "air_temp_offset_k":     0.0,
        "process_temp_offset_k": 0.0,
        "rpm_scale_factor":      1.0,
        "torque_scale_factor":   1.0,
        "tool_wear_offset_min":  0.0,
    }

    # Valid ranges for each feature (for validation)
    VALID_RANGES = {
        "Air_temperature_K_":     (295.0, 305.0),
        "Process_temperature_K_": (305.0, 315.0),
        "Rotational_speed_rpm_":  (1000.0, 2500.0),
        "Torque_Nm_":             (3.0, 80.0),
        "Tool_wear_min_":         (0.0, 250.0),
        "ambient_temp_C":         (10.0, 45.0),
        "factory_load_pct":       (50.0, 100.0),
        "humidity_pct":           (20.0, 90.0),
    }

    def map_raw_reading(self, raw: dict) -> dict:
        """
        Convert raw sensor reading to model-ready feature dict.

        Parameters:
            raw (dict): Raw sensor reading with keys:
                - air_temp_c:       Air temperature in Celsius
                - process_temp_c:   Process temperature in Celsius
                - rot_speed_rpm:    Rotational speed in RPM
                - torque_nm:        Torque in Newton-metres
                - tool_wear_min:    Tool wear in minutes
                - machine_type:     Machine type (L/M/H)
                - ambient_temp_c:   Ambient temperature in Celsius
                - factory_load_pct: Factory load percentage
                - humidity_pct:     Humidity percentage

        Returns:
            dict: Model-ready feature dict with correct units
        """
        mapped = {
            "air_temp_k":          celsius_to_kelvin(raw["air_temp_c"]) + self.CALIBRATION["air_temp_offset_k"],
            "process_temp_k":      celsius_to_kelvin(raw["process_temp_c"]) + self.CALIBRATION["process_temp_offset_k"],
            "rot_speed_rpm":       raw["rot_speed_rpm"] * self.CALIBRATION["rpm_scale_factor"],
            "torque_nm":           raw["torque_nm"] * self.CALIBRATION["torque_scale_factor"],
            "tool_wear_min":       raw["tool_wear_min"] + self.CALIBRATION["tool_wear_offset_min"],
            "machine_type":        raw["machine_type"],
            "ambient_temp_c":      raw["ambient_temp_c"],
            "factory_load_pct":    raw["factory_load_pct"],
            "humidity_pct":        raw["humidity_pct"],
        }
        return mapped

    def validate_reading(self, mapped: dict) -> tuple:
        """
        Validate that mapped sensor readings are within expected ranges.

        Returns:
            tuple: (is_valid: bool, errors: list of str)
        """
        errors = []
        check_map = {
            "Air_temperature_K_":     mapped["air_temp_k"],
            "Process_temperature_K_": mapped["process_temp_k"],
            "Rotational_speed_rpm_":  mapped["rot_speed_rpm"],
            "Torque_Nm_":             mapped["torque_nm"],
            "Tool_wear_min_":         mapped["tool_wear_min"],
            "ambient_temp_C":         mapped["ambient_temp_c"],
            "factory_load_pct":       mapped["factory_load_pct"],
            "humidity_pct":           mapped["humidity_pct"],
        }
        for feat, val in check_map.items():
            low, high = self.VALID_RANGES[feat]
            if not (low <= val <= high):
                errors.append(
                    f"{feat} = {val:.2f} is out of range [{low}, {high}]"
                )
        return (len(errors) == 0), errors

    def map_and_validate(self, raw: dict) -> tuple:
        """
        Map raw reading and validate in one step.

        Returns:
            tuple: (mapped: dict, is_valid: bool, errors: list)
        """
        mapped = self.map_raw_reading(raw)
        is_valid, errors = self.validate_reading(mapped)
        return mapped, is_valid, errors


# =================================================================
# QUICK TEST
# =================================================================
if __name__ == "__main__":
    mapper = SensorMapper()

    # Example raw reading (as it might come from ESP32 or real sensor)
    raw_reading = {
        "air_temp_c":       26.95,     # Will convert to ~300.1 K
        "process_temp_c":   36.85,     # Will convert to ~310.0 K
        "rot_speed_rpm":    1551.0,
        "torque_nm":        42.8,
        "tool_wear_min":    0.0,
        "machine_type":     "M",
        "ambient_temp_c":   28.0,
        "factory_load_pct": 75.0,
        "humidity_pct":     60.0,
    }

    mapped, is_valid, errors = mapper.map_and_validate(raw_reading)

    print("=== Sensor Mapping Test ===")
    print(f"Raw air temp:     {raw_reading['air_temp_c']}°C")
    print(f"Mapped air temp:  {mapped['air_temp_k']:.2f} K")
    print(f"Raw process temp: {raw_reading['process_temp_c']}°C")
    print(f"Mapped proc temp: {mapped['process_temp_k']:.2f} K")
    print(f"\nValid: {is_valid}")
    if errors:
        print("Errors:", errors)
    else:
        print("✅ All readings within valid ranges!")