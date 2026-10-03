"""Upload the saved local experiments and register the selected model in Staging.

Set MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD in your environment.
Run: python -m a3.publish
No other student's experiments or registered models are changed.
"""
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "20")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
import mlflow
from mlflow import MlflowClient
import pandas as pd
from a3.train import ROOT, LOCAL_URI, REMOTE_URI


def publish():
    if not os.environ.get("MLFLOW_TRACKING_USERNAME") or not os.environ.get("MLFLOW_TRACKING_PASSWORD"):
        raise RuntimeError("Set MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD first")
    summary = json.loads((ROOT / "data/a3_best_model_summary.json").read_text())
    student_id = summary["student_id"]
    experiment_name = f"{student_id}-a3"
    local = MlflowClient(tracking_uri=LOCAL_URI)
    remote = MlflowClient(tracking_uri=REMOTE_URI)
    mlflow.set_tracking_uri(REMOTE_URI)
    mlflow.set_experiment(experiment_name)
    experiment = remote.get_experiment_by_name(experiment_name)
    run_map = {}
    results = pd.read_csv(ROOT / "data/a3_experiment_results.csv")
    for local_id in results.run_id:
        original = local.get_run(local_id)
        # Re-running after a connection failure resumes only this student's run.
        matches = remote.search_runs([experiment.experiment_id],
                                     filter_string=f"tags.local_source_run = '{local_id}'")
        if matches and matches[0].data.tags.get("upload_complete") == "true":
            run_map[local_id] = matches[0].info.run_id
            continue
        with mlflow.start_run(run_id=matches[0].info.run_id if matches else None,
                              run_name=original.data.tags.get("mlflow.runName")) as run:
            mlflow.set_tags({"local_source_run": local_id, "training_location": "local",
                             "upload_complete": "false"})
            mlflow.log_params(original.data.params)
            for key in original.data.metrics:
                for metric in local.get_metric_history(local_id, key):
                    mlflow.log_metric(key, metric.value, step=metric.step)
            with tempfile.TemporaryDirectory() as temp:
                # Download only model artifacts. Never log Cars.csv or input rows.
                model_path = local.download_artifacts(local_id, "model", temp)
                mlflow.log_artifacts(model_path, artifact_path="model")
            if local_id == summary["best_config"]["run_id"]:
                mlflow.set_tag("selected_by", "highest_cv_macro_f1")
                mlflow.log_dict(summary["test_report"], "test_report.json")
            mlflow.set_tag("upload_complete", "true")
            run_map[local_id] = run.info.run_id
    best_remote = run_map[summary["best_config"]["run_id"]]
    name = f"{student_id}-a3-model"
    existing = remote.search_model_versions(f"name = '{name}'")
    versions = [v for v in existing if v.run_id == best_remote]
    version = versions[0] if versions else mlflow.register_model(f"runs:/{best_remote}/model", name)
    # Stages are deprecated in modern MLflow, but the assignment explicitly
    # requires Staging. Do not silently substitute an alias or archive versions.
    remote.transition_model_version_stage(name, version.version, "Staging", archive_existing_versions=False)
    actual = remote.get_model_version(name, version.version)
    if actual.current_stage != "Staging":
        raise RuntimeError("The registry did not confirm the required Staging state")
    # Verify the remote model can actually be downloaded and loaded for serving.
    loaded = mlflow.pyfunc.load_model(f"models:/{name}/{version.version}")
    from a3.data import FEATURES
    synthetic = pd.DataFrame([dict(year=2017., km_driven=40000., owner=1., mileage=20.,
                                  engine=1200., max_power=80., seats=5., brand="Maruti",
                                  fuel="Petrol", seller_type="Individual", transmission="Manual")], columns=FEATURES)
    if loaded.predict(synthetic).shape != (1,):
        raise RuntimeError("Remote model failed the output-shape check")
    receipt = dict(tracking_uri=REMOTE_URI, experiment_id=experiment.experiment_id,
                   run_ids=run_map, best_run_id=best_remote, registered_model=name,
                   version=version.version, stage=actual.current_stage, remote_load_verified=True)
    (ROOT / "data/a3_remote_receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    publish()
