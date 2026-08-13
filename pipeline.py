import subprocess
import sys

def run_command(cmd):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error executing {' '.join(cmd)}")
        sys.exit(result.returncode)

if __name__ == '__main__':
    commands = [
        ["python", "src/preprocess.py"],
        ["python", "src/optimize_xgboost.py"],
        ["python", "src/train_catboost.py"],
        ["python", "src/find_ensemble_weights.py"],
        ["python", "src/ensemble.py"],
        ["python", "src/portfolio_solver.py"]
    ]

    for cmd in commands:
        run_command(cmd)

    print("Pipeline executed successfully!")
