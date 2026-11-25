"""POST /predict : succès, format de sortie, valeurs aux bornes, exemples de l'interface."""
import pytest

from tests.conftest import GENRES, VALID_PAYLOAD

UI_EXAMPLES = {
    "piano_calme": {"danceability": 0.3, "energy": 0.1, "loudness": -22, "speechiness": 0.04,
                    "acousticness": 0.95, "instrumentalness": 0.9, "liveness": 0.1,
                    "valence": 0.3, "tempo": 80},
    "metal": {"danceability": 0.35, "energy": 0.95, "loudness": -4, "speechiness": 0.07,
              "acousticness": 0.02, "instrumentalness": 0.1, "liveness": 0.2,
              "valence": 0.3, "tempo": 150},
    "rap": {"danceability": 0.8, "energy": 0.65, "loudness": -7, "speechiness": 0.3,
            "acousticness": 0.15, "instrumentalness": 0.0, "liveness": 0.15,
            "valence": 0.5, "tempo": 95},
}

LOWER_BOUNDS = {"danceability": 0, "energy": 0, "loudness": -60, "speechiness": 0,
                "acousticness": 0, "instrumentalness": 0, "liveness": 0, "valence": 0, "tempo": 30}
UPPER_BOUNDS = {"danceability": 1, "energy": 1, "loudness": 5, "speechiness": 1,
                "acousticness": 1, "instrumentalness": 1, "liveness": 1, "valence": 1, "tempo": 250}


def test_valid_prediction_has_expected_format(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"top3", "model_name", "model_version"}
    assert body["model_name"] == "genre-classifier"
    assert body["model_version"] == "test"
    assert len(body["top3"]) == 3


def test_top3_is_sorted_distinct_and_valid(client):
    top3 = client.post("/predict", json=VALID_PAYLOAD).json()["top3"]
    probas = [item["proba"] for item in top3]
    genres = [item["genre"] for item in top3]
    assert probas == sorted(probas, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in probas)
    assert sum(probas) <= 1.0 + 1e-6
    assert len(set(genres)) == 3
    assert set(genres) <= set(GENRES)


@pytest.mark.parametrize("name", UI_EXAMPLES)
def test_ui_examples_are_accepted(client, name):
    r = client.post("/predict", json=UI_EXAMPLES[name])
    assert r.status_code == 200
    assert len(r.json()["top3"]) == 3


@pytest.mark.parametrize("bounds", [LOWER_BOUNDS, UPPER_BOUNDS], ids=["lower", "upper"])
def test_values_at_the_bounds_are_accepted(client, bounds):
    r = client.post("/predict", json=bounds)
    assert r.status_code == 200


def test_integers_are_accepted_for_float_fields(client):
    payload = {**VALID_PAYLOAD, "tempo": 120, "energy": 1, "danceability": 0}
    assert client.post("/predict", json=payload).status_code == 200