import os
import optuna
from typing import Optional

def setup_optuna_study(
    study_name: str,
    db_name: Optional[str] = None,
    direction: str = "minimize"
) -> optuna.Study:
    """
    Setup standard directories and create/load an Optuna study.
    If db_name is provided, it uses SQLite storage and creates
    necessary directories. Otherwise, it creates an in-memory study.
    """
    if db_name:
        os.makedirs("./experiments", exist_ok=True)
        os.makedirs("./models", exist_ok=True)
        storage_url = f"sqlite:///{os.path.abspath(f'./experiments/{db_name}')}"

        study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            storage=storage_url,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(
            direction=direction,
            study_name=study_name
        )

    return study
