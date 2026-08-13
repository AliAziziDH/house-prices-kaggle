import os
import subprocess


def run_step(command):
    print(
        f"\n=====================================\nRunning: {command}\n====================================="
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    subprocess.run(command, shell=True, check=True, env=env)


def main():
    run_step("python3 src/preprocess.py")
    run_step("python3 src/optimize_xgboost.py")
    run_step("python3 src/train_catboost.py")
    run_step("python3 src/find_ensemble_weights.py")
    run_step("python3 src/ensemble.py")
    run_step("python3 src/recommend_portfolio.py")


if __name__ == "__main__":
    main()
