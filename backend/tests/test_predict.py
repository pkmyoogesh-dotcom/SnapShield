"""API-contract tests use a stub model (see conftest). Real-model tests are skipped
until `python -m ml.train` has produced backend/models/."""
import pytest
from fastapi.testclient import TestClient

from backend import config
from backend.main import app

KEYS = {"url", "classification", "risk_score", "risk_percentage", "reasons", "contributions", "features", "inference"}


def test_response_format(client):
    r = client.post("/api/v1/predict", json={"url": "https://example.com"})
    assert r.status_code == 200 and KEYS <= set(r.json())
    assert r.json()["classification"] in {"SAFE", "SUSPICIOUS", "PHISHING"}
    assert r.json()["risk_percentage"] == round(r.json()["risk_score"] * 100)

def test_reasons_come_from_features(client):
    j = client.post("/api/v1/predict", json={"url": "http://192.168.1.1/login/verify"}).json()
    assert any("IP" in x for x in j["reasons"]) and j["classification"] != "SAFE"

@pytest.mark.parametrize("body", [{"url": ""}, {}, {"url": "not a url"}, {"url": "ftp://a.com"}, {"url": "x" * 3000}])
def test_rejects_bad_input(client, body):
    assert client.post("/api/v1/predict", json=body).status_code == 422

def test_oversized_body(client):
    assert client.post("/api/v1/predict", content=b"x" * 10000, headers={"Content-Type": "application/json"}).status_code == 413

@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="model not trained yet")
def test_real_model_loads_and_responds():
    r = TestClient(app).post("/api/v1/predict", json={"url": "https://www.wikipedia.org"})
    assert r.status_code == 200 and 0 <= r.json()["risk_score"] <= 1
