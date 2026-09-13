from __future__ import annotations

from itertools import product
from pathlib import Path
import json
import os
import sys

# Silence an unrelated MLflow GenAI helper message before importing MLflow.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
# OUTPUT_DIR = ROOT_DIR / "assignment2" / "task2"
MODEL_DIR = ROOT_DIR / "models"

from a2_regularization import LinearRegression, Lasso, NoRegularization, Ridge


RANDOM_STATE = 42
CV_EPOCHS = 12
FINAL_EPOCHS = 60
BATCH_SIZE = 128
REGULARIZATION_STRENGTH = 0.001
GRADIENT_CLIP = 10.0
NUMERIC = ["year", "km_driven", "owner", "mileage", "engine", "max_power", "seats"]
CATEGORICAL = ["brand", "fuel", "seller_type", "transmission"]


def make_regularization(name):
    """Convert the grid's readable label into the object required by the class."""
    regularizers = {
        "normal": NoRegularization,
        "lasso": lambda: Lasso(REGULARIZATION_STRENGTH),
        "ridge": lambda: Ridge(REGULARIZATION_STRENGTH),
    }
    try:
        return regularizers[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown regularization: {name}") from exc


def add_intercept(X):
    """Add the leading column of ones expected by the course regression class."""
    X = np.asarray(X, dtype=float)
    return np.column_stack([np.ones(X.shape[0]), X])


def clean_data(frame):
    # Repeat the attached Assignment 1 notebook's cleaning order exactly.
    data = frame.copy()
    data = data.loc[~data["fuel"].isin(["CNG", "LPG"])]
    data["owner"] = data["owner"].map(
        {
            "First Owner": 1,
            "Second Owner": 2,
            "Third Owner": 3,
            "Fourth & Above Owner": 4,
            "Test Drive Car": 5,
        }
    )
    for column in ["mileage", "engine", "max_power"]:
        data[column] = pd.to_numeric(
            data[column].astype("string").str.extract(r"([-+]?\d*\.?\d+)", expand=False),
            errors="coerce",
        )
    data = data.rename(columns={"name": "brand"})
    data["brand"] = data["brand"].astype("string").str.split().str[0]
    data = data.drop(columns=["torque"])
    data = data.loc[data["owner"] != 5].copy()

    # Deleted the small group of rows missing any of these four fields.
    required_columns = ["mileage", "engine", "max_power", "seats"]
    missing_mask = data[required_columns].isnull().any(axis=1)
    data = data.loc[~missing_mask].copy().reset_index(drop=True)

    # Deduplication happens after parsing and brand extraction in A1.
    return data.drop_duplicates().reset_index(drop=True).copy()


def make_preprocessor(polynomial=False):
    """Create fold-local imputation, scaling, encoding, and optional polynomial terms."""
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ]
    if polynomial:
        numeric_steps.extend(
            [
                ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
                ("polynomial_scaler", StandardScaler()),
            ]
        )
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), NUMERIC),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL,
            ),
        ]
    )


def build_grid():
    """Return the exact 4 × 2 × 3 × 2 × 3 factorial design."""
    model_specs = {
        "normal": (False, "normal"),
        "polynomial": (True, "normal"),
        "lasso": (False, "lasso"),
        "ridge": (False, "ridge"),
    }
    configurations = []
    for model_type, use_momentum, method, initialization, learning_rate in product(
        model_specs,
        [False, True],
        ["stochastic", "mini-batch", "batch"],
        ["zeros", "xavier"],
        [0.01, 0.001, 0.0001],
    ):
        polynomial, regularization = model_specs[model_type]
        configurations.append(
            {
                "model_type": model_type,
                "polynomial": polynomial,
                "regularization": regularization,
                "use_momentum": use_momentum,
                "method": method,
                "initialization": initialization,
                "learning_rate": learning_rate,
            }
        )
    assert len(configurations) == 144
    return configurations


def evaluate_config(X, y, config, splits):
    """Evaluate one configuration on the same five folds used by every run."""
    fold_rows = []
    for fold, (train_idx, validation_idx) in enumerate(splits, start=1):
        preprocessor = make_preprocessor(config["polynomial"])
        x_train = add_intercept(preprocessor.fit_transform(X.iloc[train_idx]))
        x_validation = add_intercept(preprocessor.transform(X.iloc[validation_idx]))
        y_train = y.iloc[train_idx].to_numpy()
        y_validation = y.iloc[validation_idx].to_numpy()

        target_mean = y_train.mean()
        target_scale = y_train.std()
        model = LinearRegression(
            regularization=make_regularization(config["regularization"]),
            lr=config["learning_rate"],
            method=config["method"],
            initialization=config["initialization"],
            use_momentum=config["use_momentum"],
            momentum=0.9,
            num_epochs=CV_EPOCHS,
            batch_size=BATCH_SIZE,
            random_state=RANDOM_STATE + fold,
            gradient_clip=GRADIENT_CLIP,
            cv=False,
            verbose=False,
        ).fit(x_train, (y_train - target_mean) / target_scale)

        prediction = model.predict(x_validation) * target_scale + target_mean
        fold_mse = model.mse(y_validation, prediction)
        fold_r2 = model.r2(y_validation, prediction)
        fold_rows.append({"fold": fold, "mse": fold_mse, "r2": fold_r2})
        mlflow.log_metric("fold_mse", fold_mse, step=fold)
        mlflow.log_metric("fold_r2", fold_r2, step=fold)

    fold_frame = pd.DataFrame(fold_rows)
    return {
        "cv_mse": float(fold_frame["mse"].mean()),
        "cv_mse_std": float(fold_frame["mse"].std(ddof=0)),
        "cv_r2": float(fold_frame["r2"].mean()),
        "cv_r2_std": float(fold_frame["r2"].std(ddof=0)),
    }


def fit_best_model(X_train, y_train, X_test, y_test, best):
    """Refit the selected configuration on all training data and evaluate once on test data."""
    preprocessor = make_preprocessor(bool(best["polynomial"]))
    x_train = add_intercept(preprocessor.fit_transform(X_train))
    x_test = add_intercept(preprocessor.transform(X_test))
    target_mean = float(y_train.mean())
    target_scale = float(y_train.std())

    model = LinearRegression(
        regularization=make_regularization(best["regularization"]),
        lr=float(best["learning_rate"]),
        method=best["method"],
        initialization=best["initialization"],
        use_momentum=bool(best["use_momentum"]),
        momentum=0.9,
        num_epochs=FINAL_EPOCHS,
        batch_size=BATCH_SIZE,
        random_state=RANDOM_STATE,
        gradient_clip=GRADIENT_CLIP,
        cv=False,
        verbose=False,
    ).fit(x_train, (y_train.to_numpy() - target_mean) / target_scale)

    predicted_log_price = model.predict(x_test) * target_scale + target_mean
    actual_price = np.exp(y_test.to_numpy())
    predicted_price = np.exp(predicted_log_price)
    test_metrics = {
        "test_mse_log": model.mse(y_test.to_numpy(), predicted_log_price),
        "test_r2_log": model.r2(y_test.to_numpy(), predicted_log_price),
        "test_mse_price": model.mse(actual_price, predicted_price),
        "test_rmse_price": float(np.sqrt(model.mse(actual_price, predicted_price))),
        "test_r2_price": model.r2(actual_price, predicted_price),
    }
    bundle = {
        "preprocessor": preprocessor,
        "model": model,
        "target_mean": target_mean,
        "target_scale": target_scale,
        "feature_names": preprocessor.get_feature_names_out().tolist(),
        "best_config": best,
        "test_metrics": test_metrics,
    }
    return bundle


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    data = clean_data(pd.read_csv(DATA_DIR / "Cars.csv"))
    X = data[NUMERIC + CATEGORICAL]
    # Assignment 1 modeled log selling price, so Task 2 retains the same target.
    y = np.log(data["selling_price"].astype(float))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    configurations = build_grid()
    splits = list(KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(X_train))

    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    tracking_db = DATA_DIR / "a2_mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db}")
    mlflow.set_experiment("A2 Task 2 - Based on A1 Car Price Notebook")
    rows = []

    for run_number, config in enumerate(configurations, start=1):
        # Training and validation happen inside the MLflow run, not before it.
        with mlflow.start_run(run_name=f"{config['model_type']}-{run_number:03d}"):
            mlflow.log_params(
                config
                | {
                    "folds": 5,
                    "epochs": CV_EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "regularization_strength": REGULARIZATION_STRENGTH,
                    "gradient_clip": GRADIENT_CLIP,
                    "target": "log_selling_price",
                }
            )
            metrics = evaluate_config(X_train, y_train, config, splits)
            mlflow.log_metrics(metrics)

        row = {**config, **metrics}
        rows.append(row)
        print(
            f"[{run_number:03d}/144] R2={metrics['cv_r2']:.4f} "
            f"MSE={metrics['cv_mse']:.5f} {config}"
        )

    results = pd.DataFrame(rows).sort_values(
        ["cv_r2", "cv_mse"], ascending=[False, True]
    )
    results.to_csv(DATA_DIR / "a2_experiment_results.csv", index=False)
    best = results.iloc[0].to_dict()
    bundle = fit_best_model(X_train, y_train, X_test, y_test, best)

    joblib.dump(bundle, MODEL_DIR / "best_a2_model.joblib")
    summary = {"best_config": best, "test_metrics": bundle["test_metrics"]}
    (DATA_DIR / "a2_best_model_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print("BEST", json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
