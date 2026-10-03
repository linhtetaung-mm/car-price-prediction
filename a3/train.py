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

from a3.data import FEATURES, PRICE_EDGES, PRICE_LABELS, clean_data, make_preprocessor, price_classes
from a3.metrics import classification_report, confusion_matrix
from a3.model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "st127132-a3"
REMOTE_URI = "https://mlflow.ml.brain.cs.ait.ac.th"
LOCAL_URI = f"sqlite:///{ROOT / 'data/a3_mlflow.db'}"


def metric_values(y, predicted, prefix):
    report = classification_report(y, predicted)
    return {f"{prefix}_accuracy": report["accuracy"],
            **{f"{prefix}_{kind}_{metric.replace('-', '_')}": report[f"{kind} avg"][metric]
               for kind in ["macro", "weighted"] for metric in ["precision", "recall", "f1-score"]}}


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
    raw_path = ROOT / "data/Cars.csv"
    data = clean_data(pd.read_csv(raw_path))
    X, y = data[FEATURES], price_classes(data.selling_price)
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=42)
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train, y_train))
    # Each fold has its own imputer/scaler/encoder. Cache transformations only;
    # every candidate gets newly initialized weights and the same fold indices.
    prepared = []
    for tr, va in folds:
        prep = make_preprocessor()
        prepared.append((prep.fit_transform(X_train.iloc[tr]), y_train[tr],
                         prep.transform(X_train.iloc[va]), y_train[va]))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(f"{student_id}-a3")
    rows, fitted = [], {}
    for lr in [.03, .1, .3]:
        for l2 in [0., .0001, .001, .01]:
            name = f"{'unregularized' if l2 == 0 else 'ridge'}-lr{lr}-l2{l2}"
            params = dict(lr=lr, l2=l2, num_epochs=epochs, folds=5, random_state=42,
                          optimizer="batch", objective="mean_cross_entropy_plus_l2",
                          price_edges="0,300000,600000,1000000,inf", dataset_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest())
            with mlflow.start_run(run_name=name) as run:
                mlflow.log_params(params)
                scores = []
                for fold, (xt, yt, xv, yv) in enumerate(prepared, 1):
                    model = LogisticRegression(lr=lr, l2=l2, num_epochs=epochs).fit(xt, yt)
                    values = metric_values(yv, model.predict(xv), "validation")
                    mlflow.log_metrics(values, step=fold)
                    scores.append(values)
                metrics = {key.replace("validation_", "cv_"): float(np.mean([s[key] for s in scores]))
                           for key in scores[0]}
                metrics["cv_macro_f1_std"] = float(np.std([s["validation_macro_f1_score"] for s in scores]))
                mlflow.log_metrics(metrics)
                pipeline = Pipeline([("preprocessor", make_preprocessor()),
                                     ("classifier", LogisticRegression(lr=lr, l2=l2, num_epochs=epochs))])
                pipeline.fit(X_train, y_train)
                # Save via the stable run-artifact API, compatible with older course servers.
                with tempfile.TemporaryDirectory() as temp:
                    path = Path(temp) / "model"
                    save_mlflow_model(pipeline, path)
                    mlflow.log_artifacts(str(path), artifact_path="model")
                run_id = run.info.run_id
                fitted[run_id] = pipeline
                rows.append(dict(run_id=run_id, **params, **metrics))
                print(f"{name}: CV macro F1={metrics['cv_macro_f1_score']:.4f}", flush=True)
    results = pd.DataFrame(rows).sort_values(["cv_macro_f1_score", "cv_accuracy", "l2"], ascending=[False, False, True])
    best = results.iloc[0].to_dict()
    pipeline = fitted[best["run_id"]]
    # The test set is evaluated once, AFTER selecting hyperparameters by CV.
    predicted = pipeline.predict(X_test)
    report = classification_report(y_test, predicted)
    test_metrics = metric_values(y_test, predicted, "test")
    counts = lambda values: {str(c): int(np.sum(values == c)) for c in range(4)}
    summary = dict(student_id=student_id, tracking_uri=tracking_uri, experiment_name=f"{student_id}-a3",
                   cleaned_rows=len(data), train_rows=len(train_idx), test_rows=len(test_idx),
                   train_class_counts=counts(y_train), test_class_counts=counts(y_test),
                   price_labels=PRICE_LABELS, best_config=best, test_metrics=test_metrics,
                   test_report=report, majority_baseline_accuracy=float(np.mean(y_test == np.bincount(y_train).argmax())),
                   remote_status="logged" if tracking_uri == REMOTE_URI else "pending upload",
                   selection="5-fold stratified CV macro F1; test not used for selection")
    results.to_csv(ROOT / "data/a3_experiment_results.csv", index=False)
    (ROOT / "data/a3_best_model_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(confusion_matrix(y_test, predicted), index=range(4), columns=range(4)).to_csv(ROOT / "data/a3_confusion_matrix.csv")
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
