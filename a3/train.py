"""Run reproducible experiments: python -m a3.train (local MLflow by default)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from a3.data import FEATURES, PRICE_LABELS, clean_data, make_preprocessor, price_classes
from a3.metrics import classification_report, confusion_matrix
from a3.model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
REMOTE_URI = "https://mlflow.ml.brain.cs.ait.ac.th"
LOCAL_URI = f"sqlite:///{ROOT / 'data/a3_mlflow.db'}"


RANDOM_STATE = 42
LEARNING_RATES = [0.03, 0.1, 0.3]
L2_STRENGTHS = [0.0, 0.0001, 0.001, 0.01]


def metric_values(y_true, y_pred, prefix):
    """Give the report values simple names for the MLflow runs table."""
    report = classification_report(y_true, y_pred)
    metrics = {f"{prefix}_accuracy": report["accuracy"]}

    for average in ["macro", "weighted"]:
        for metric in ["precision", "recall", "f1-score"]:
            metric_name = metric.replace("-", "_")
            key = f"{prefix}_{average}_{metric_name}"
            metrics[key] = report[f"{average} avg"][metric]
    return metrics


def prepare_folds(X_train, y_train):
    """Fit preprocessing on the training part of each fold."""
    cross_validation = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    prepared_folds = []
    for train_index, validation_index in cross_validation.split(X_train, y_train):
        preprocessor = make_preprocessor()
        x_train = preprocessor.fit_transform(X_train.iloc[train_index])
        x_validation = preprocessor.transform(X_train.iloc[validation_index])
        prepared_folds.append((
            x_train, y_train[train_index],
            x_validation, y_train[validation_index],
        ))
    return prepared_folds


def evaluate_config(prepared_folds, lr, l2, epochs):
    """Train a fresh model on every fold and average the validation scores."""
    fold_scores = []
    for fold, (x_train, y_train, x_validation, y_validation) in enumerate(
        prepared_folds, start=1
    ):
        model = LogisticRegression(lr=lr, l2=l2, num_epochs=epochs)
        model.fit(x_train, y_train)
        prediction = model.predict(x_validation)
        scores = metric_values(y_validation, prediction, "validation")
        mlflow.log_metrics(scores, step=fold)
        fold_scores.append(scores)

    fold_results = pd.DataFrame(fold_scores)
    metrics = {}
    for column in fold_results.columns:
        name = column.replace("validation_", "cv_")
        metrics[name] = float(fold_results[column].mean())
    metrics["cv_macro_f1_std"] = float(
        fold_results["validation_macro_f1_score"].std(ddof=0)
    )
    return metrics


def fit_model(X_train, y_train, lr, l2, epochs):
    """Keep preprocessing and the classifier together for the web application."""
    pipeline = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("classifier", LogisticRegression(lr=lr, l2=l2, num_epochs=epochs)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def class_counts(y):
    counts = {}
    for label in range(4):
        counts[str(label)] = int(np.sum(y == label))
    return counts


def save_mlflow_model(pipeline, directory):
    # Only the fitted model and Python code are included. No dataset, raw input
    # examples, autologging, or log_input calls are used, as required by the brief.
    from mlflow.models import ModelSignature
    from mlflow.types.schema import ColSpec, Schema
    from a3.data import NUMERIC, CATEGORICAL
    signature = ModelSignature(
        inputs=Schema([ColSpec("double", c) for c in NUMERIC]
                      + [ColSpec("string", c) for c in CATEGORICAL]),
        outputs=Schema([ColSpec("long")]))
    mlflow.sklearn.save_model(
        pipeline, str(directory), code_paths=[str(ROOT / "a3")], signature=signature,
        serialization_format="cloudpickle",
        pip_requirements=["numpy==2.5.2", "pandas==3.0.5", "scikit-learn==1.9.0",
                          "joblib==1.5.3", "mlflow==3.16.0"])


def run_experiments(tracking_uri=LOCAL_URI, student_id="st127132", epochs=600):
    # 1. Use the same cleaned data and price bands as the notebook.
    raw_path = ROOT / "data/Cars.csv"
    data = clean_data(pd.read_csv(raw_path))
    X = data[FEATURES]
    y = price_classes(data["selling_price"])
    train_index, test_index = train_test_split(
        np.arange(len(y)), test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_train, y_train = X.iloc[train_index], y[train_index]
    X_test, y_test = X.iloc[test_index], y[test_index]

    # These transformed folds can be reused because the input data is the same
    # for every configuration. The models themselves are always fitted again.
    prepared_folds = prepare_folds(X_train, y_train)
    dataset_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(f"{student_id}-a3")
    rows = []
    fitted_models = {}

    # 2. Compare learning rates and ridge strengths using training data only.
    for lr in LEARNING_RATES:
        for l2 in L2_STRENGTHS:
            model_type = "unregularized" if l2 == 0 else "ridge"
            run_name = f"{model_type}-lr{lr}-l2{l2}"
            params = {
                "lr": lr,
                "l2": l2,
                "num_epochs": epochs,
                "folds": 5,
                "random_state": RANDOM_STATE,
                "optimizer": "batch",
                "objective": "mean_cross_entropy_plus_l2",
                "price_edges": "0,300000,600000,1000000,inf",
                "dataset_sha256": dataset_hash,
            }
            with mlflow.start_run(run_name=run_name) as run:
                mlflow.log_params(params)
                metrics = evaluate_config(prepared_folds, lr, l2, epochs)
                mlflow.log_metrics(metrics)
                pipeline = fit_model(X_train, y_train, lr, l2, epochs)

                # Save each candidate so the selected model can be registered later.
                with tempfile.TemporaryDirectory() as temporary_directory:
                    model_path = Path(temporary_directory) / "model"
                    save_mlflow_model(pipeline, model_path)
                    mlflow.log_artifacts(str(model_path), artifact_path="model")

                run_id = run.info.run_id
                fitted_models[run_id] = pipeline
                rows.append({"run_id": run_id, **params, **metrics})
                print(
                    f"{run_name}: CV macro F1={metrics['cv_macro_f1_score']:.4f}",
                    flush=True,
                )

    # 3. Select the best CV result, then evaluate that model on the test set.
    results = pd.DataFrame(rows).sort_values(
        ["cv_macro_f1_score", "cv_accuracy", "l2"],
        ascending=[False, False, True],
    )
    best = results.iloc[0].to_dict()
    pipeline = fitted_models[best["run_id"]]
    prediction = pipeline.predict(X_test)
    report = classification_report(y_test, prediction)
    test_metrics = metric_values(y_test, prediction, "test")
    majority_class = np.bincount(y_train).argmax()

    summary = {
        "student_id": student_id,
        "tracking_uri": tracking_uri,
        "experiment_name": f"{student_id}-a3",
        "cleaned_rows": len(data),
        "train_rows": len(train_index),
        "test_rows": len(test_index),
        "train_class_counts": class_counts(y_train),
        "test_class_counts": class_counts(y_test),
        "price_labels": PRICE_LABELS,
        "best_config": best,
        "test_metrics": test_metrics,
        "test_report": report,
        "majority_baseline_accuracy": float(np.mean(y_test == majority_class)),
        "remote_status": "logged" if tracking_uri == REMOTE_URI else "pending upload",
        "selection": "5-fold stratified CV macro F1; test not used for selection",
    }

    # 4. Save the files used by the notebook and application.
    results.to_csv(ROOT / "data/a3_experiment_results.csv", index=False)
    summary_path = ROOT / "data/a3_best_model_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    matrix = pd.DataFrame(
        confusion_matrix(y_test, prediction), index=range(4), columns=range(4)
    )
    matrix.to_csv(ROOT / "data/a3_confusion_matrix.csv")
    joblib.dump(pipeline, ROOT / "models/best_a3_model.joblib")

    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.set_tag("selected_by", "highest_cv_macro_f1")
        mlflow.log_metrics(test_metrics)
        mlflow.log_dict(report, "test_report.json")
        mlflow.log_dict(summary, "summary.json")

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default=LOCAL_URI)
    parser.add_argument("--student-id", default="st127132")
    parser.add_argument("--epochs", type=int, default=600)
    args = parser.parse_args()
    run_experiments(args.tracking_uri, args.student_id, args.epochs)
