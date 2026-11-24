"""
Train a model on a dataset version, evaluate it on the held-out set
and log everything to MLflow (DagsHub). The model is registered as a new
version of "genre-classifier" but NOT promoted (see ml/promote.py).

Run: python -m ml.train --version v1 --model logreg
"""
import argparse
import os
import subprocess

import mlflow
import pandas as pd
import yaml
from dotenv import load_dotenv
from matplotlib.figure import Figure
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml import config

load_dotenv()  # reads .env locally, does nothing in GitHub Actions

EVAL_PATH = "data/eval/eval.csv"


def get_git_sha():
    if os.getenv("GITHUB_SHA"):
        return os.getenv("GITHUB_SHA")
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def get_dvc_md5(dvc_file):
    with open(dvc_file) as f:
        return yaml.safe_load(f)["outs"][0]["md5"]


def build_model(model_type):
    if model_type == "logreg":
        clf = LogisticRegression(max_iter=1000)
    elif model_type == "rf":
        clf = RandomForestClassifier(n_estimators=300, random_state=config.SEED, n_jobs=-1)
    else:
        raise ValueError(f"unknown model type: {model_type}")
    # scaler is fitted on the training data only (inside the pipeline)
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def load_data(version):
    train = pd.read_csv(f"data/{version}/train.csv")
    eval_df = pd.read_csv(EVAL_PATH)
    return (train[config.FEATURES], train[config.LABEL],
            eval_df[config.FEATURES], eval_df[config.LABEL])


def evaluate(model, X, y):
    preds = model.predict(X)
    metrics = {
        "macro_f1": f1_score(y, preds, average="macro"),
        "accuracy": accuracy_score(y, preds),
    }
    return metrics, preds


def train(version, model_type):
    X_train, y_train, X_eval, y_eval = load_data(version)

    mlflow.set_experiment(config.EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"{model_type}-{version}") as run:
        mlflow.set_tags({
            "git_sha": get_git_sha(),
            "dataset_version": version,
            "dvc_md5": get_dvc_md5(f"data/{version}.dvc"),
            "eval_dvc_md5": get_dvc_md5("data/eval.dvc"),
            "model_type": model_type,
        })

        model = build_model(model_type)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_params({k: v for k, v in model.named_steps["clf"].get_params().items()
                           if isinstance(v, (int, float, str, bool)) or v is None})

        model.fit(X_train, y_train)
        metrics, preds = evaluate(model, X_eval, y_eval)
        mlflow.log_metrics(metrics)
        print(f"{model_type} on {version}: macro_f1={metrics['macro_f1']:.4f} "
              f"accuracy={metrics['accuracy']:.4f}")

        # evaluation artifacts
        report = classification_report(y_eval, preds)
        mlflow.log_text(report, "classification_report.txt")
        fig = Figure(figsize=(8, 8))  # no window needed, just save the plot
        ax = fig.subplots()
        ConfusionMatrixDisplay.from_predictions(y_eval, preds, ax=ax, xticks_rotation=45)
        fig.tight_layout()
        mlflow.log_figure(fig, "confusion_matrix.png")

        # log + register the model (new version, no alias yet)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=config.MODEL_NAME,
            input_example=X_eval.head(3),
        )

        client = MlflowClient()
        mv = client.search_model_versions(f"run_id='{run.info.run_id}'")[0]
        client.set_model_version_tag(config.MODEL_NAME, mv.version, "dataset_version", version)
        client.set_model_version_tag(config.MODEL_NAME, mv.version, "macro_f1",
                                     f"{metrics['macro_f1']:.4f}")
        client.set_model_version_tag(config.MODEL_NAME, mv.version, "git_sha", get_git_sha())
        print(f"registered {config.MODEL_NAME} version {mv.version}")

    return run.info.run_id, mv.version, metrics["macro_f1"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, choices=config.DATASET_VERSIONS)
    parser.add_argument("--model", default=config.BASELINE_MODEL, choices=["logreg", "rf"])
    args = parser.parse_args()
    train(args.version, args.model)


if __name__ == "__main__":
    main()
