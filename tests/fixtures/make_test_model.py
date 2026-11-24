"""fake model"""
import argparse
import shutil
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

GENRES = ["classical", "country", "hip-hop", "jazz", "metal", "pop", "reggae", "techno"]
FEATURES = ["danceability", "energy", "loudness", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo"]


def build_model():
    rng = np.random.default_rng(42)
    n = 40 * len(GENRES)
    X = pd.DataFrame(rng.random((n, len(FEATURES))), columns=FEATURES)
    X["loudness"] = X["loudness"] * 65 - 60      # -60 à 5
    X["tempo"] = X["tempo"] * 220 + 30           # 30 à 250
    y = np.repeat(GENRES, 40)
    model = Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=1000))])
    model.fit(X, y)
    return model, X


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    out = Path(p.parse_args().out)
    if out.exists():
        shutil.rmtree(out)
    model, X = build_model()
    mlflow.sklearn.save_model(model, str(out), input_example=X.head(3))
    print(f"Test model saved to {out}")