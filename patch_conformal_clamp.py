with open('src/conformal.py', 'r') as f:
    content = f.read()

# We'll just enforce that upper_bound is at least y_pred_point, or we clamp y_pred_point as well!
# "Keep the standard boundaries physical clamping strictly at [$42,000, $525,000]." implies clamping all predictions.
# Let's clamp y_pred_point, lower_bound, and upper_bound.

new_content = content.replace(
    '    upper_bound = np.clip(upper_bound, a_min=None, a_max=max_physical_price)',
    '    upper_bound = np.clip(upper_bound, a_min=None, a_max=max_physical_price)\n    upper_bound = np.maximum(upper_bound, y_pred_point)'
)

with open('src/conformal.py', 'w') as f:
    f.write(new_content)
