"""Chargement du modèle approuvé : alias champion -> version concrète, readiness, reload."""
from fastapi.testclient import TestClient

from app.main import app
from app.model_loader import ModelHolder
from tests.conftest import ADMIN_TOKEN, MODEL_NAME, VALID_PAYLOAD

ADMIN = {"X-Admin-Token": ADMIN_TOKEN}


def test_loads_champion_even_if_a_newer_version_exists(test_registry):
    # Le registry contient v1 (champion) et v2 (plus récente, sans alias)
    versions = test_registry.search_model_versions(f"name='{MODEL_NAME}'")
    assert {str(mv.version) for mv in versions} == {"1", "2"}
    holder = ModelHolder()
    assert holder.try_load() is True
    assert holder.version == "1"
    assert holder.ready is True


def test_model_info_reports_the_loaded_version(registry_client):
    info = registry_client.get("/model-info").json()
    assert info["model_name"] == MODEL_NAME
    assert info["model_alias"] == "champion"
    assert info["model_version"] == "1"
    assert info["loaded_at"] is not None
    assert registry_client.get("/ready").json() == {"status": "ready", "model_version": "1"}
    assert registry_client.post("/predict", json=VALID_PAYLOAD).json()["model_version"] == "1"


def test_not_ready_without_champion(test_registry, monkeypatch):
    test_registry.delete_registered_model_alias(MODEL_NAME, "champion")
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200     # le processus vit
        assert c.get("/ready").status_code == 503       # mais pas de modèle approuvé
        assert c.get("/model-info").json()["model_version"] is None
        assert c.post("/predict", json=VALID_PAYLOAD).status_code == 503


def test_not_ready_when_registry_is_unreachable(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:////nonexistent-dir/none.db")
    holder = ModelHolder()
    assert holder.try_load() is False
    assert holder.ready is False
    assert holder.last_error


def test_reload_picks_up_a_moved_alias(registry_client, test_registry):
    assert registry_client.get("/model-info").json()["model_version"] == "1"
    test_registry.set_registered_model_alias(MODEL_NAME, "champion", "2")
    r = registry_client.post("/admin/reload", headers=ADMIN)
    assert r.status_code == 200
    assert r.json()["model_version"] == "2"
    assert registry_client.get("/model-info").json()["model_version"] == "2"
    assert registry_client.post("/predict", json=VALID_PAYLOAD).json()["model_version"] == "2"


def test_failed_reload_keeps_the_previous_model(registry_client, test_registry):
    test_registry.delete_registered_model_alias(MODEL_NAME, "champion")
    assert registry_client.post("/admin/reload", headers=ADMIN).status_code == 503
    assert registry_client.get("/ready").status_code == 200
    assert registry_client.post("/predict", json=VALID_PAYLOAD).json()["model_version"] == "1"


def test_reload_requires_the_admin_token(registry_client):
    assert registry_client.post("/admin/reload").status_code == 403
    assert registry_client.post("/admin/reload", headers={"X-Admin-Token": "wrong"}).status_code == 403


def test_reload_is_disabled_without_admin_token_configured(client):
    assert client.post("/admin/reload", headers=ADMIN).status_code == 403


def test_local_model_uri_is_used_in_ci_mode(client):
    assert client.get("/model-info").json()["model_version"] == "test"