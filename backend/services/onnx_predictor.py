"""ONNX Runtime inference (CPUExecutionProvider). ONNX by itself does NOT imply NPU acceleration."""
from pathlib import Path

import numpy as np
import onnxruntime as ort


class OnnxModel:
    def __init__(self, path: Path):
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict_proba(self, X) -> np.ndarray:
        out = self.session.run(None, {self.input_name: np.asarray(X, dtype=np.float32)})
        return np.asarray(out[1])  # (n, 2) probabilities; zipmap disabled at export
