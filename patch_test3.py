with open('tests/test_pipeline.py', 'r') as f:
    content = f.read()

# Since the prompt said: "Keep the standard boundaries physical clamping strictly at [$42,000, $525,000]."
# If the point prediction is ABOVE 525,000, then upper_bound is also above 525,000 because we enforced np.maximum(upper_bound, point_prediction).
# Should we clamp the POINT PREDICTION to 525000 too? Yes, "clamping strictly at [$42,000, $525,000]" implies nothing should be > 525000.
# Let's check `src/conformal.py` again. We should clip y_pred_point too!

content_conf = """import numpy as np


def compute_non_conformity_scores(y_true, y_pred):
    return np.abs(y_true - y_pred)


def compute_empirical_quantile(scores, alpha=0.05):
    n = len(scores)
    q_level = min(100.0, 100.0 * (1 - alpha) * (n + 1) / n)
    q = np.percentile(scores, q_level)
    return q


def compute_prediction_intervals(y_pred_point_log, q, min_physical_price=42000.0, max_physical_price=525000.0):
    lower_log = y_pred_point_log - q
    upper_log = y_pred_point_log + q

    y_pred_point = np.expm1(y_pred_point_log)
    lower_bound = np.expm1(lower_log)
    upper_bound = np.expm1(upper_log)

    # Clip all predictions to physical bounds
    y_pred_point = np.clip(y_pred_point, a_min=min_physical_price, a_max=max_physical_price)
    lower_bound = np.clip(lower_bound, a_min=min_physical_price, a_max=max_physical_price)
    upper_bound = np.clip(upper_bound, a_min=min_physical_price, a_max=max_physical_price)

    upper_bound = np.maximum(upper_bound, y_pred_point)

    return y_pred_point, lower_bound, upper_bound
"""
with open('src/conformal.py', 'w') as f:
    f.write(content_conf)
