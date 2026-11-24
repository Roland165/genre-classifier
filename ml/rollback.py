"""
Roll back production to a previously approved model version.
After this, redeploy / refresh the app so it loads the restored version.

Run: python -m ml.rollback --version 1
"""
import argparse
import sys

from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from ml import config
from ml.promote import promote

load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="approved version to restore")
    args = parser.parse_args()

    client = MlflowClient()
    try:
        mv = client.get_model_version(config.MODEL_NAME, args.version)
    except MlflowException:
        sys.exit(f"version {args.version} does not exist")

    # only allow versions that passed the quality gates before
    if mv.tags.get("approved") != "true":
        sys.exit(f"version {args.version} was never approved, refusing rollback")

    promote(args.version)
    print(f"rollback done, champion is version {args.version}")


if __name__ == "__main__":
    main()
