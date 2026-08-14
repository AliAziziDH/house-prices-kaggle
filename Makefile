SHELL := /bin/bash

.PHONY: all preprocess optimize_xgb train_catboost optimize_lightgbm find_weights make_submissions recommend_portfolio clean

all: preprocess optimize_xgb train_catboost train_linear find_weights make_submissions recommend_portfolio

preprocess:
	PYTHONPATH=. PYTHONPATH=. python src/preprocess.py

optimize_xgb:
	PYTHONPATH=. PYTHONPATH=. python src/optimize_xgboost.py

train_catboost:
	PYTHONPATH=. PYTHONPATH=. python src/train_catboost.py

train_linear:
	PYTHONPATH=. PYTHONPATH=. python src/train_linear_models.py

optimize_lightgbm:
	python experiments/train_lightgbm.py

find_weights:
	PYTHONPATH=. PYTHONPATH=. python src/find_ensemble_weights.py

make_submissions:
	# (a) Leaderboard weights: XGB 0.1667, CAT 0.1665
	PYTHONPATH=. PYTHONPATH=. python src/ensemble.py

dashboard:
	streamlit run src/ui/app.py

recommend_portfolio:
	PYTHONPATH=. PYTHONPATH=. python src/portfolio_solver.py

clean:
	rm -rf processed_data/* models/* submissions/* experiments/archive/*
