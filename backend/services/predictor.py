"""Local model loading (scikit-learn joblib, or ONNX Runtime if enabled)."""
from __future__ import annotations

import json

import joblib
import numpy as np

from backend import config
from backend.services.feature_extractor import FEATURE_NAMES


class ModelBundle:
    def __init__(self, model, metadata: dict, backend_name: str):
        self.model, self.metadata, self.backend_name = model, metadata, backend_name
        self.baseline: dict[str, float] = metadata["safe_baseline"]
        self.feature_names: list[str] = metadata["feature_names"]

    @classmethod
    def load(cls) -> "ModelBundle":
        if not config.METADATA_PATH.exists():
            raise FileNotFoundError("Model not trained yet. Run: python -m ml.train")
        metadata = json.loads(config.METADATA_PATH.read_text())
        if metadata["feature_names"] != FEATURE_NAMES:
            raise RuntimeError("Model features do not match extractor; retrain the model.")
        if config.USE_ONNX and config.ONNX_PATH.exists():
            from backend.services.onnx_predictor import OnnxModel
            return cls(OnnxModel(config.ONNX_PATH), metadata, "onnxruntime")
        if not config.MODEL_PATH.exists():
            raise FileNotFoundError("Model file missing. Run: python -m ml.train")
        return cls(joblib.load(config.MODEL_PATH), metadata, "scikit-learn")

    def predict_proba(self, X) -> np.ndarray:
        """Probability of the PHISHING class for each row."""
        X = np.asarray(X, dtype=np.float64)
        out = self.model.predict_proba(X)
        return out[:, 1] if getattr(out, "ndim", 1) == 2 else out


def classify(score: float) -> str:
    if score >= config.PHISHING_THRESHOLD:
        return "PHISHING"
    if score >= config.SUSPICIOUS_THRESHOLD:
        return "SUSPICIOUS"
    return "SAFE"
