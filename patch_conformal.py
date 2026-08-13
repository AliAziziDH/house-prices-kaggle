import re

with open('src/conformal.py', 'r') as f:
    content = f.read()

# Add max_physical_price
content = content.replace('def compute_prediction_intervals(y_pred_point_log, q, min_physical_price=42000.0):', 'def compute_prediction_intervals(y_pred_point_log, q, min_physical_price=42000.0, max_physical_price=525000.0):')
content = content.replace('lower_bound = np.clip(lower_bound, a_min=min_physical_price, a_max=None)', 'lower_bound = np.clip(lower_bound, a_min=min_physical_price, a_max=None)\n    upper_bound = np.clip(upper_bound, a_min=None, a_max=max_physical_price)')

with open('src/conformal.py', 'w') as f:
    f.write(content)
