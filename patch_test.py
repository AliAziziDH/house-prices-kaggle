import re

with open('tests/test_pipeline.py', 'r') as f:
    content = f.read()

content = content.replace('assert (df["SalePrice_Lower"] >= 42000.0).all()', 'assert (df["SalePrice_Lower"] >= 42000.0).all()\n        # Upper bound clamped to max $525000\n        assert (df["SalePrice_Upper"] <= 525000.0).all()')

with open('tests/test_pipeline.py', 'w') as f:
    f.write(content)
