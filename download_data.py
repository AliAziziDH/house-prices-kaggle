import os
import shutil

import kagglehub

# Download latest version of the competition files
path = kagglehub.competition_download("house-prices-advanced-regression-techniques")
print("Downloaded to:", path)

# Copy files to our local data/ folder
os.makedirs("data", exist_ok=True)
for file in os.listdir(path):
    shutil.copy(os.path.join(path, file), os.path.join("data", file))
print("✅ Raw files successfully copied to data/")
