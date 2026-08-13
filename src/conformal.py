import numpy as np


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
