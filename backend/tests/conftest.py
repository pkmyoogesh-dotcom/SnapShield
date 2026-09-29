import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app, get_bundle
from backend.services.feature_extractor import FEATURE_NAMES


class StubModel:
    """Deterministic stand-in used ONLY to test API plumbing (not a real classifier)."""
    def predict_proba(self, X):
        X = np.asarray(X)
        kw, ip, ln = (FEATURE_NAMES.index(n) for n in ("keyword_count", "has_ip_host", "url_length"))
        p = np.clip(0.03 + 0.25 * X[:, kw] + 0.5 * X[:, ip] + 0.002 * X[:, ln], 0, 0.99)
        return np.column_stack([1 - p, p])


class StubBundle:
    from backend.services.predictor import ModelBundle
    predict_proba = ModelBundle.predict_proba
    def __init__(self):
        self.model, self.backend_name = StubModel(), "stub"
        self.baseline = {n: 0.0 for n in FEATURE_NAMES}
        self.metadata = {"model_type": "stub"}


@pytest.fixture()
def client():
    app.dependency_overrides[get_bundle] = lambda: StubBundle()
    yield TestClient(app)
    app.dependency_overrides.clear()
