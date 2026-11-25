"""Entrées invalides : l'API doit répondre 422 (jamais 500, jamais une prédiction)."""
import pytest

from tests.conftest import VALID_PAYLOAD


def _with(**changes):
    return {**VALID_PAYLOAD, **changes}


def _without(field):
    return {k: v for k, v in VALID_PAYLOAD.items() if k != field}


OUT_OF_RANGE = [
    pytest.param(_with(energy=1.5), id="energy>1"),
    pytest.param(_with(energy=-0.01), id="energy<0"),
    pytest.param(_with(danceability=2), id="danceability>1"),
    pytest.param(_with(tempo=-10), id="tempo negatif"),
    pytest.param(_with(tempo=0), id="tempo=0 (invalide)"),
    pytest.param(_with(tempo=29.9), id="tempo<30"),
    pytest.param(_with(tempo=250.1), id="tempo>250"),
    pytest.param(_with(loudness=-60.1), id="loudness<-60"),
    pytest.param(_with(loudness=5.1), id="loudness>5"),
]

BAD_TYPES = [
    pytest.param(_with(energy="fort"), id="chaine non numerique"),
    pytest.param(_with(tempo=None), id="null"),
    pytest.param(_with(valence=[0.5]), id="liste"),
    pytest.param(_with(energy="0.5"), id="nombre en chaine (strict)"),
    pytest.param(_with(energy=True), id="booleen (strict)"),
]


@pytest.mark.parametrize("payload", OUT_OF_RANGE)
def test_out_of_range_values_return_422(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("payload", BAD_TYPES)
def test_wrong_types_return_422(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("field", list(VALID_PAYLOAD))
def test_missing_field_returns_422(client, field):
    r = client.post("/predict", json=_without(field))
    assert r.status_code == 422
    assert field in str(r.json()["detail"])


def test_extra_field_returns_422(client):
    assert client.post("/predict", json=_with(foo=1)).status_code == 422


def test_empty_and_malformed_bodies_return_422(client):
    assert client.post("/predict", json={}).status_code == 422
    assert client.post("/predict").status_code == 422
    r = client.post("/predict", content="pas du json",
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 422


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_numbers_are_rejected(client, token):
    raw = '{"danceability":0.5,"energy":%s,"loudness":-5,"speechiness":0.05,' \
          '"acousticness":0.1,"instrumentalness":0,"liveness":0.1,"valence":0.5,"tempo":120}' % token
    r = client.post("/predict", content=raw, headers={"Content-Type": "application/json"})
    assert r.status_code == 422