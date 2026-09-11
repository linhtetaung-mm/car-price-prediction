"""Run the complete Assignment 2 grid, log it to MLflow, and save the best model."""

from __future__ import annotations

from itertools import product
from pathlib import Path
import json
import os

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from linear_regression import LinearRegression


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
RANDOM_STATE = 42
NUMERIC = ["year", "km_driven", "owner", "mileage", "engine", "max_power", "seats"]
CATEGORICAL = ["brand", "fuel", "seller_type", "transmission"]


def clean_data(frame):
    data = frame.drop_duplicates().copy()
    data = data.loc[~data["fuel"].isin(["CNG", "LPG"])]
    data = data.loc[data["owner"] != "Test Drive Car"].copy()
    data["owner"] = data["owner"].map({
        "First Owner": 1, "Second Owner": 2, "Third Owner": 3,
        "Fourth & Above Owner": 4,
    })
    for column in ["mileage", "engine", "max_power"]:
        data[column] = pd.to_numeric(
            data[column].astype("string").str.extract(r"([-+]?\d*\.?\d+)", expand=False),
            errors="coerce",
        )
    data["brand"] = data["name"].astype("string").str.split().str[0]
    return data.drop(columns=["name", "torque"]).reset_index(drop=True)


def make_preprocessor(polynomial=False):
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if polynomial:
        numeric_steps.extend([
            ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
            ("polynomial_scaler", StandardScaler()),
        ])
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), NUMERIC),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL),
    ])


def evaluate_config(X, y, config, splits):
    fold_mse, fold_r2 = [], []
    for fold, (train_idx, val_idx) in enumerate(splits):
        preprocessor = make_preprocessor(config["model_type"] == "polynomial")
        x_train = preprocessor.fit_transform(X.iloc[train_idx])
        x_val = preprocessor.transform(X.iloc[val_idx])
        y_train, y_val = y.iloc[train_idx].to_numpy(), y.iloc[val_idx].to_numpy()
        target_mean, target_scale = y_train.mean(), y_train.std()
        model = LinearRegression(
            regularization=config["regularization"],
            learning_rate=config["learning_rate"], method=config["method"],
            initialization=config["initialization"], use_momentum=config["use_momentum"],
            momentum=0.9, regularization_strength=0.001,
            num_epochs=12, batch_size=128, random_state=RANDOM_STATE + fold,
        ).fit(x_train, (y_train - target_mean) / target_scale)
        prediction = model.predict(x_val) * target_scale + target_mean
        fold_mse.append(model.mse(y_val, prediction))
        fold_r2.append(model.r2(y_val, prediction))
    return float(np.mean(fold_mse)), float(np.mean(fold_r2)), float(np.std(fold_r2))


def main():
    data = clean_data(pd.read_csv(PROJECT_ROOT / "data" / "Cars.csv"))
    X = data[NUMERIC + CATEGORICAL]
    y = np.log(data["selling_price"].astype(float))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE,
    )

    # "polynomial" changes the numeric feature map; the other three select the penalty.
    model_specs = {
        "normal": (False, "normal"),
        "polynomial": (True, "normal"),
        "lasso": (False, "lasso"),
        "ridge": (False, "ridge"),
    }
    configs = []
    for model_type, use_momentum, method, initialization, learning_rate in product(
        model_specs, [False, True], ["stochastic", "mini-batch", "batch"],
        ["zeros", "xavier"], [0.01, 0.001, 0.0001],
    ):
        polynomial, regularization = model_specs[model_type]
        configs.append({
            "model_type": model_type, "polynomial": polynomial,
            "regularization": regularization, "use_momentum": use_momentum,
            "method": method, "initialization": initialization,
            "learning_rate": learning_rate,
        })

    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    tracking_db = ROOT / "mlflow_a2.db"
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db}")
    mlflow.set_experiment("A2 Car Price From-Scratch Regression")
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(cv.split(X_train))
    rows = []

    for number, config in enumerate(configs, 1):
        mse, r2, r2_std = evaluate_config(X_train, y_train, config, splits)
        row = {**config, "cv_mse": mse, "cv_r2": r2, "cv_r2_std": r2_std}
        rows.append(row)
        with mlflow.start_run(run_name=f"{config['model_type']}-{number:03d}"):
            mlflow.log_params(config | {"epochs": 12, "folds": 5, "batch_size": 128,
                                               "gradient_clip": 10.0})
            mlflow.log_metrics({"cv_mse": mse, "cv_r2": r2, "cv_r2_std": r2_std})
        print(f"[{number:03d}/{len(configs)}] R2={r2:.4f} MSE={mse:.5f} {config}")

    results = pd.DataFrame(rows).sort_values(["cv_r2", "cv_mse"], ascending=[False, True])
    results.to_csv(ROOT / "experiment_results.csv", index=False)
    best = results.iloc[0].to_dict()

    preprocessor = make_preprocessor(bool(best["polynomial"]))
    x_train = preprocessor.fit_transform(X_train)
    x_test = preprocessor.transform(X_test)
    target_mean, target_scale = float(y_train.mean()), float(y_train.std())
    model = LinearRegression(
        regularization=best["regularization"], learning_rate=float(best["learning_rate"]),
        method=best["method"], initialization=best["initialization"],
        use_momentum=bool(best["use_momentum"]), momentum=0.9,
        regularization_strength=0.001, num_epochs=60, batch_size=128,
        random_state=RANDOM_STATE,
    ).fit(x_train, (y_train.to_numpy() - target_mean) / target_scale)
    predicted_log = model.predict(x_test) * target_scale + target_mean
    test_metrics = {
        "test_mse_log": model.mse(y_test, predicted_log),
        "test_r2_log": model.r2(y_test, predicted_log),
    }
    feature_names = preprocessor.get_feature_names_out().tolist()
    bundle = {
        "preprocessor": preprocessor, "model": model,
        "target_mean": target_mean, "target_scale": target_scale,
        "feature_names": feature_names, "best_config": best,
        "test_metrics": test_metrics,
    }
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(bundle, ROOT / "models" / "car_price_prediction_a2.joblib")
    (ROOT / "best_model_summary.json").write_text(
        json.dumps({"best_config": best, "test_metrics": test_metrics}, indent=2, default=str)
    )
    print("BEST", json.dumps({"best_config": best, "test_metrics": test_metrics}, indent=2, default=str))


if __name__ == "__main__":
    main()
