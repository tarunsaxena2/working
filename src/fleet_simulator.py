"""
fleet_simulator.py — Virtual Machine Fleet Simulator
Vaibhav Gautam — Dashboard & Integration
Week 2 Day 1 Task

Generates 3-4 virtual machines, each with a distinct
sensor profile, for the Fleet Overview dashboard page.
"""

import numpy as np
import pandas as pd

# =================================================================
# VIRTUAL MACHINE PROFILES
# =================================================================
MACHINES = [
    {
        "id":          "MACHINE-01",
        "name":        "CNC Mill A",
        "type":        "M",
        "type_enc":    1,
        "location":    "Floor 1 — Bay A",
        "description": "High-load milling machine — priority asset",
        "profile": {
            "air_temp_mean":    300.5, "air_temp_std":    1.2,
            "proc_temp_mean":   311.0, "proc_temp_std":   0.8,
            "rot_speed_mean":   1600,  "rot_speed_std":   120,
            "torque_mean":      55.0,  "torque_std":      8.0,
            "tool_wear_mean":   180,   "tool_wear_std":   30,
        },
    },
    {
        "id":          "MACHINE-02",
        "name":        "Lathe B",
        "type":        "L",
        "type_enc":    0,
        "location":    "Floor 1 — Bay B",
        "description": "Standard lathe — moderate usage",
        "profile": {
            "air_temp_mean":    299.5, "air_temp_std":    1.0,
            "proc_temp_mean":   309.5, "proc_temp_std":   0.7,
            "rot_speed_mean":   1400,  "rot_speed_std":   100,
            "torque_mean":      38.0,  "torque_std":      6.0,
            "tool_wear_mean":   80,    "tool_wear_std":   20,
        },
    },
    {
        "id":          "MACHINE-03",
        "name":        "Press C",
        "type":        "H",
        "type_enc":    2,
        "location":    "Floor 2 — Bay A",
        "description": "Heavy-duty press — high wear rate",
        "profile": {
            "air_temp_mean":    301.5, "air_temp_std":    1.5,
            "proc_temp_mean":   312.0, "proc_temp_std":   1.0,
            "rot_speed_mean":   1800,  "rot_speed_std":   150,
            "torque_mean":      68.0,  "torque_std":      10.0,
            "tool_wear_mean":   220,   "tool_wear_std":   25,
        },
    },
    {
        "id":          "MACHINE-04",
        "name":        "Grinder D",
        "type":        "M",
        "type_enc":    1,
        "location":    "Floor 2 — Bay B",
        "description": "Grinding machine — recently serviced",
        "profile": {
            "air_temp_mean":    299.0, "air_temp_std":    0.8,
            "proc_temp_mean":   309.0, "proc_temp_std":   0.6,
            "rot_speed_mean":   1350,  "rot_speed_std":   80,
            "torque_mean":      32.0,  "torque_std":      5.0,
            "tool_wear_mean":   25,    "tool_wear_std":   10,
        },
    },
]


def generate_reading(machine: dict, seed: int = None) -> dict:
    """
    Generate a single simulated sensor reading for a machine.

    Parameters:
        machine (dict): Machine profile from MACHINES list
        seed (int): Random seed for reproducibility

    Returns:
        dict: Sensor reading ready for /predict API
    """
    rng = np.random.default_rng(seed)
    p   = machine["profile"]

    return {
        "Air_temperature_K":     float(rng.normal(p["air_temp_mean"],  p["air_temp_std"])),
        "Process_temperature_K": float(rng.normal(p["proc_temp_mean"], p["proc_temp_std"])),
        "Rotational_speed_rpm":  float(rng.normal(p["rot_speed_mean"], p["rot_speed_std"])),
        "Torque_Nm":             float(rng.normal(p["torque_mean"],     p["torque_std"])),
        "Tool_wear_min":         float(np.clip(rng.normal(p["tool_wear_mean"], p["tool_wear_std"]), 0, 250)),
        "Type_enc":              machine["type_enc"],
        "ambient_temp_C":        float(rng.normal(28, 5)),
        "factory_load_pct":      float(rng.uniform(50, 100)),
        "humidity_pct":          float(rng.normal(60, 10)),
    }


def get_fleet_readings(seed: int = None) -> list:
    """
    Generate one reading per machine in the fleet.

    Returns:
        list: List of dicts with machine info + sensor reading
    """
    readings = []
    for i, machine in enumerate(MACHINES):
        reading = generate_reading(machine, seed=(seed + i) if seed else None)
        readings.append({
            "machine_id":   machine["id"],
            "machine_name": machine["name"],
            "machine_type": machine["type"],
            "location":     machine["location"],
            "description":  machine["description"],
            "reading":      reading,
        })
    return readings


def get_fleet_dataframe(seed: int = None) -> pd.DataFrame:
    """
    Return fleet readings as a flat DataFrame for display.

    Returns:
        pd.DataFrame: One row per machine with sensor values
    """
    readings = get_fleet_readings(seed)
    rows = []
    for r in readings:
        row = {"Machine": r["machine_name"], "Type": r["machine_type"],
               "Location": r["location"]}
        row.update(r["reading"])
        rows.append(row)
    return pd.DataFrame(rows)


# =================================================================
# QUICK TEST
# =================================================================
if __name__ == "__main__":
    print("=== Fleet Simulator Test ===")
    readings = get_fleet_readings(seed=42)
    for r in readings:
        print(f"\n{r['machine_id']} — {r['machine_name']}")
        print(f"  Location: {r['location']}")
        print(f"  Torque:   {r['reading']['Torque_Nm']:.2f} Nm")
        print(f"  Tool wear:{r['reading']['Tool_wear_min']:.0f} min")
    print("\n✅ Fleet simulator working!")