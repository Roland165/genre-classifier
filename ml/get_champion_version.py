"""
Print the version number of the current champion (used by smoke tests).

Run: python -m ml.get_champion_version
"""
import sys

from dotenv import load_dotenv
from mlflow import MlflowClient

from ml.promote import get_champion_version

load_dotenv()

version = get_champion_version(MlflowClient())
if version is None:
    sys.exit("no champion")
print(version)
