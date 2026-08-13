import re

with open('src/optimize_xgboost.py', 'r') as f:
    content = f.read()

content = re.sub(r'    best_params_no_rs = \{k: v for k, v in best_params\.items\(\) if k != "random_state"\}\n', '', content)

with open('src/optimize_xgboost.py', 'w') as f:
    f.write(content)
