"""
Continuous training pipeline (called by the CT GitHub workflow,
after `dvc pull` and `python -m ml.validate_data`).

Steps:
  1. evaluate the current champion on the held-out set
  2. decide if we retrain (ml/decide.py)
  3. train a candidate and compare it with the champion on the same eval set
  4. promote it or keep the champion
Everything is logged in an MLflow run named "ct-<version>".

Run: python -m ml.ct_pipeline --version v2 --model rf [--force]
"""
import argparse
import os

import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow import MlflowClient
from sklearn.metrics import f1_score

from ml import config
from ml.decide import should_promote, should_retrain
from ml.promote import get_champion_version, promote
from ml.train import train

load_dotenv()


def evaluate_champion(client):
    """Return (version, macro_f1, dataset_version) of the champion, or Nones."""
    version = get_champion_version(client)
    if version is None:
        return None, None, None

    mv = client.get_model_version(config.MODEL_NAME, version)
    model = mlflow.sklearn.load_model(f"models:/{config.MODEL_NAME}/{version}")
    eval_df = pd.read_csv("data/eval/eval.csv")
    preds = model.predict(eval_df[config.FEATURES])
    f1 = f1_score(eval_df[config.LABEL], preds, average="macro")
    return version, f1, mv.tags.get("dataset_version")


def write_github_output(**values):
    # lets the next workflow steps read the result (steps.ct.outputs.xxx)
    path = os.getenv("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as f:
            f.writelines(f"{key}={value}\n" for key, value in values.items())


def run(version, model_type, force=False):
    client = MlflowClient()
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    # 1. evaluate the champion
    champ_version, champ_f1, champ_dataset = evaluate_champion(client)
    if champ_version:
        print(f"champion: version {champ_version}, macro_f1={champ_f1:.4f}, data={champ_dataset}")
    else:
        print("no champion yet")

    # 2. decide
    retrain, retrain_reason = should_retrain(champ_f1, champ_dataset, version, force=force)
    print(f"retrain: {retrain} ({retrain_reason})")

    cand_version, cand_f1 = None, None
    promoted, promote_reason = False, "no retraining"

    if retrain:
        # 3. train + compare (train() evaluates on the same eval set)
        _, cand_version, cand_f1 = train(version, model_type)
        promoted, promote_reason = should_promote(cand_f1, champ_f1)
        print(f"promote: {promoted} ({promote_reason})")

        # 4. promote or keep the champion
        if promoted:
            promote(cand_version)
        else:
            client.set_model_version_tag(config.MODEL_NAME, cand_version, "approved", "false")
            client.set_model_version_tag(config.MODEL_NAME, cand_version,
                                         "rejection_reason", promote_reason)
            print(f"keeping champion version {champ_version}")

    new_champion = get_champion_version(client)

    # log the whole decision in its own run
    with mlflow.start_run(run_name=f"ct-{version}"):
        mlflow.set_tags({
            "run_type": "continuous_training",
            "dataset_version": version,
            "model_type": model_type,
            "forced": str(force),
            "retrain_decision": str(retrain),
            "retrain_reason": retrain_reason,
            "promotion_decision": str(promoted),
            "promotion_reason": promote_reason,
            "champion_version_before": str(champ_version),
            "candidate_version": str(cand_version),
            "champion_version_after": str(new_champion),
        })
        if champ_f1 is not None:
            mlflow.log_metric("champion_macro_f1", champ_f1)
        if cand_f1 is not None:
            mlflow.log_metric("candidate_macro_f1", cand_f1)

    write_github_output(
        retrained=str(retrain).lower(),
        promoted=str(promoted).lower(),
        champion_version=new_champion,
    )
    print(f"done, champion is now version {new_champion}")
    return promoted, new_champion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, choices=config.DATASET_VERSIONS)
    parser.add_argument("--model", default=config.CANDIDATE_MODEL, choices=["logreg", "rf"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(args.version, args.model, args.force)


if __name__ == "__main__":
    main()
