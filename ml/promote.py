"""
Make a registered model version the production model:
- tag it as approved
- move the "previous" alias to the current champion (useful for rollback)
- move the "champion" alias to the new version

Run (first release, by hand): python -m ml.promote --version 1
Later the CT pipeline calls promote() directly.
"""
import argparse

from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from ml import config

load_dotenv()


def get_champion_version(client):
    try:
        return client.get_model_version_by_alias(config.MODEL_NAME, config.CHAMPION_ALIAS).version
    except MlflowException:
        return None  # no champion yet


def promote(version):
    client = MlflowClient()
    old = get_champion_version(client)

    client.set_model_version_tag(config.MODEL_NAME, version, "approved", "true")
    if old is not None and str(old) != str(version):
        client.set_registered_model_alias(config.MODEL_NAME, config.PREVIOUS_ALIAS, old)
    client.set_registered_model_alias(config.MODEL_NAME, config.CHAMPION_ALIAS, version)

    print(f"champion: {old} -> {version}")
    return old


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="model version number, e.g. 1")
    args = parser.parse_args()
    promote(args.version)


if __name__ == "__main__":
    main()
