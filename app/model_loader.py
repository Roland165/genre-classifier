"""Chargement du modèle : alias champion du registry MLflow, ou modèle local (CI/tests)."""
import logging
import os
import threading
import time
from datetime import UTC, datetime

import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient

logger = logging.getLogger("model_loader")

FEATURES = ["danceability", "energy", "loudness", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo"]

SAMPLE_INPUT = {"danceability": 0.62, "energy": 0.81, "loudness": -5.2,
                "speechiness": 0.05, "acousticness": 0.10, "instrumentalness": 0.0,
                "liveness": 0.12, "valence": 0.55, "tempo": 124.0}

RETRY_SECONDS = 30


class ModelHolder:
    def __init__(self):
        self.name = os.getenv("MODEL_NAME", "genre-classifier")
        self.alias = os.getenv("MODEL_ALIAS", "champion")
        self.model = None
        self.version = None
        self.loaded_at = None
        self.ready = False
        self.last_error = None
        self._lock = threading.Lock()

    def load(self):
        """Charge le modèle. Lève une exception en cas d'échec (l'ancien modèle reste en place)."""
        uri = os.getenv("MODEL_URI")
        if uri:  # CI / tests : modèle local
            model = mlflow.sklearn.load_model(uri)
            version = os.getenv("MODEL_VERSION", "test")
        else:    # production : registry -> alias résolu en version concrète
            client = MlflowClient()
            mv = client.get_model_version_by_alias(self.name, self.alias)
            version = str(mv.version)
            model = mlflow.sklearn.load_model(f"models:/{self.name}/{version}")

        self._check(model)  # vérifie que le modèle prédit vraiment
        with self._lock:
            self.model, self.version = model, version
            self.loaded_at = datetime.now(UTC).isoformat()
            self.ready, self.last_error = True, None
        logger.info("Model %s loaded (version %s)", self.name, version)

    def try_load(self):
        try:
            self.load()
            return True
        except Exception as exc:  # noqa: BLE001 - tout échec de chargement doit rester non fatal
            self.last_error = str(exc)
            logger.error("Model load failed: %s", exc)
            return False

    def start_retry_thread(self):
        """Réessaie toutes les 30 s tant que le modèle n'est pas chargé."""
        def loop():
            while True:
                time.sleep(RETRY_SECONDS)
                if not self.ready:
                    self.try_load()
        threading.Thread(target=loop, daemon=True).start()

    @staticmethod
    def _check(model):
        df = pd.DataFrame([SAMPLE_INPUT], columns=FEATURES)
        proba = model.predict_proba(df)[0]
        if len(proba) != len(model.classes_):
            raise ValueError("predict_proba output does not match model.classes_")

    def predict_top3(self, features: dict):
        with self._lock:
            model, version = self.model, self.version
        df = pd.DataFrame([features], columns=FEATURES)  # ordre des colonnes garanti
        proba = model.predict_proba(df)[0]
        ranked = sorted(zip(model.classes_, proba), key=lambda p: p[1], reverse=True)[:3]
        return [{"genre": str(g), "proba": round(float(p), 4)} for g, p in ranked], version