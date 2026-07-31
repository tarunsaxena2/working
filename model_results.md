# \# Model Results — Final Confirmed Metrics

# 

# \## Best Model Configuration

# | Parameter | Value |

# |---|---|

# | Model | LightGBM + SMOTE |

# | n\_estimators | 500 |

# | learning\_rate | 0.1 |

# | num\_leaves | 15 |

# | scale\_pos\_weight | Removed |

# | CV Strategy | 5-Fold Stratified |

# 

# \## Final Metrics (Holdout Test Set)

# | Metric | Value | Status |

# |---|---|---|

# | Macro F1 | 0.8501 | ✅ KPI Met (≥0.85) |

# | Precision | 0.8233 | ✅ |

# | Recall | 0.8825 | ✅ |

# 

# \## Notes

# \- SMOTE applied only inside training folds — no data leakage

# \- External context features included (ambient\_temp\_C, factory\_load\_pct, humidity\_pct)

# \- scale\_pos\_weight removed after tuning — SMOTE alone handles imbalance sufficiently

# \- Metrics verified by both Tarun Saxena and Vaibhav Gautam — confirmed match ✅

