"""Fixtures communes : modèle local de test, client HTTP, registry MLflow SQLite temporaire.

Aucun test ne touche le réseau (pas de DagsHub, pas d'Azure) : la CI peut tous les lancer.
"""
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.main import app
from app.model_loader import FEATURES

GENRES = ["classical", "country", "hip-hop", "jazz", "metal", "pop", "reggae", "techno"]
MODEL_NAME = "genre-classifier"
ADMIN_TOKEN = "secret-test-token"

VALID_PAYLOAD = {
    "danceability": 0.62, "energy": 0.81, "loudness": -5.2, "speechiness": 0.05,
    "acousticness": 0.10, "instrumentalness": 0.0, "liveness": 0.12,
    "valence": 0.55, "tempo": 124.0,
}


def build_model(seed: int = 0):
    """Petit modèle entraîné sur des données aléatoires, avec les 8 genres."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.random((200, len(FEATURES))), columns=FEATURES)
    y = np.tile(GENRES, 25)
    model = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=200))])
    return model.fit(X, y)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Isole chaque test de l'environnement du développeur (.env, shell...)."""
    for var in ("MODEL_URI", "MODEL_VERSION", "MODEL_NAME", "MODEL_ALIAS", "ADMIN_TOKEN"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def local_model_dir(tmp_path):
    path = tmp_path / "test-model"
    mlflow.sklearn.save_model(build_model(), str(path))
    return str(path)


@pytest.fixture
def client(monkeypatch, local_model_dir):
    """App démarrée avec un modèle local (comme en CI avec MODEL_URI)."""
    monkeypatch.setenv("MODEL_URI", local_model_dir)
    monkeypatch.setenv("MODEL_VERSION", "test")
    with TestClient(app) as c:  # `with` exécute le lifespan (chargement du modèle)
        yield c


@pytest.fixture
def test_registry(tmp_path, monkeypatch):
    """Registry MLflow SQLite temporaire : 2 versions de genre-classifier, champion -> v1.

    La v2 est plus récente mais n'a pas l'alias : l'app ne doit pas la charger.
    """
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.create_experiment("tests", artifact_location=(tmp_path / "artifacts").as_uri())
    mlflow.set_experiment("tests")
    for seed in (1, 2):
        with mlflow.start_run():
            mlflow.sklearn.log_model(build_model(seed), "model", registered_model_name=MODEL_NAME)
    registry = MlflowClient()
    registry.set_registered_model_alias(MODEL_NAME, "champion", "1")
    return registry


@pytest.fixture
def registry_client(test_registry, monkeypatch):
    """App démarrée sans MODEL_URI : elle résout l'alias champion dans le registry de test."""
    monkeypatch.setenv("ADMIN_TOKEN", ADMIN_TOKEN)
    with TestClient(app) as c:
        yield c