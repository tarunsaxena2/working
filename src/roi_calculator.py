"""
roi_calculator.py — ROI / Cost-Savings Estimator
Contextual Predictive Maintenance — IoT Edge AI

Estimates the rupee value of catching a machine failure early,
based on downtime cost, avoided downtime hours, and model confidence.

Usage:
    from src.roi_calculator import estimate_savings
"""


def estimate_savings(
    downtime_cost_per_hour: float,
    hours_downtime_avoided: float,
    failure_probability: float
) -> dict:
    """
    Estimate the rupee savings from catching a predicted failure early.

    Parameters:
        downtime_cost_per_hour (float): Cost in Rs. per hour of unplanned downtime
        hours_downtime_avoided (float): Estimated hours of downtime avoided by
            acting on this prediction instead of waiting for an unplanned failure
        failure_probability (float): Model's predicted failure probability (0.0 - 1.0)

    Returns:
        dict: {
            'estimated_savings': float,
            'downtime_cost_per_hour': float,
            'hours_downtime_avoided': float,
            'failure_probability': float
        }

    Raises:
        ValueError: if any input is negative, or failure_probability is out of [0, 1]
    """
    if downtime_cost_per_hour < 0:
        raise ValueError("downtime_cost_per_hour cannot be negative")
    if hours_downtime_avoided < 0:
        raise ValueError("hours_downtime_avoided cannot be negative")
    if not (0.0 <= failure_probability <= 1.0):
        raise ValueError("failure_probability must be between 0 and 1")

    estimated_savings = downtime_cost_per_hour * hours_downtime_avoided * failure_probability

    return {
        'estimated_savings': round(estimated_savings, 2),
        'downtime_cost_per_hour': downtime_cost_per_hour,
        'hours_downtime_avoided': hours_downtime_avoided,
        'failure_probability': failure_probability
    }


if __name__ == "__main__":
    # Quick sanity test
    result = estimate_savings(
        downtime_cost_per_hour=5000,
        hours_downtime_avoided=8,
        failure_probability=0.87
    )
    print("=== ROI Calculator Test ===")
    print(f"Downtime cost/hour: Rs. {result['downtime_cost_per_hour']}")
    print(f"Hours avoided: {result['hours_downtime_avoided']}")
    print(f"Failure probability: {result['failure_probability']}")
    print(f"Estimated savings: Rs. {result['estimated_savings']}")
