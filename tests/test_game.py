"""Tests for the "Beat the AI" game: page, track list, and every track accepted by /predict."""
import json
from pathlib import Path

from tests.conftest import GENRES

TRACKS_FILE = Path("app/game_tracks.json")


def load_tracks():
    return json.loads(TRACKS_FILE.read_text(encoding="utf-8"))


def test_game_page_is_served(client):
    r = client.get("/game")
    assert r.status_code == 200
    assert "Beat the AI" in r.text


def test_track_list_has_every_genre():
    tracks = load_tracks()
    assert len(tracks) >= len(GENRES)
    assert {t["genre"] for t in tracks} == set(GENRES)
    for t in tracks:
        assert t["track_id"] and t["name"] and t["artists"]


def test_tracks_endpoint_returns_the_file(client):
    r = client.get("/game/tracks")
    assert r.status_code == 200
    assert r.json() == load_tracks()


def test_every_game_track_is_a_valid_prediction_input(client):
    # if one track breaks the data contract, the game would fail during the demo
    for t in load_tracks():
        r = client.post("/predict", json=t["features"])
        assert r.status_code == 200, t["name"]
        assert len(r.json()["top3"]) == 3
