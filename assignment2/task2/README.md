# Assignment 2 — Task 2

This folder contains the complete MLflow-tracked comparison requested in Task 2.

From the repository root, regenerate the experiment with:

```bash
venv/bin/python assignment2/task2/run_task2_mlflow.py
```

Then open its MLflow dashboard with:

```bash
cd assignment2/task2
../../venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow_task2_a1.db --port 5000
```

The runner follows the attached Assignment 1 notebook's cleaning order, deleted-row policy, feature
set, log-price target, 80/20 split with random state 42, and preprocessing. Preprocessing is fitted
separately inside every cross-validation fold. Each of the 144 configurations is evaluated inside an
active MLflow run and logs its parameters, fold metrics, mean MSE, and mean R².

The database is a generated local artifact and is ignored by Git. The final CSV, best-model bundle,
executed notebook, and MLflow screenshot are the portable submission artifacts.
