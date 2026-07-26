base_features = [
    "Type_enc",

    "Air temperature [K]_rolling_mean",
    "Process temperature [K]_rolling_mean",
    "Rotational speed [rpm]_rolling_mean",
    "Torque [Nm]_rolling_mean",
    "Tool wear [min]_rolling_mean",

    "Air temperature [K]_rolling_std",
    "Process temperature [K]_rolling_std",
    "Rotational speed [rpm]_rolling_std",
    "Torque [Nm]_rolling_std",
    "Tool wear [min]_rolling_std",

    "Air temperature [K]_rolling_var",
    "Process temperature [K]_rolling_var",
    "Rotational speed [rpm]_rolling_var",
    "Torque [Nm]_rolling_var",
    "Tool wear [min]_rolling_var"
]
ext_features = base_features + [
    "ambient_temp_C",
    "factory_load_pct",
    "humidity_pct"
]