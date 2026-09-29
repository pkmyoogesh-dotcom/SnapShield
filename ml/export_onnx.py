"""Export to ONNX and compare with the sklearn model. Run: python -m ml.export_onnx"""
import json

import joblib
import numpy as np
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from backend import config
from backend.services.feature_extractor import FEATURE_NAMES
from backend.services.onnx_predictor import OnnxModel


def main():
    model = joblib.load(config.MODEL_PATH)
    onx = convert_sklearn(model, initial_types=[("input", FloatTensorType([None, len(FEATURE_NAMES)]))],
                          options={id(model): {"zipmap": False}} if not hasattr(model, "steps") else {id(model.steps[-1][1]): {"zipmap": False}},
                          target_opset={"": 15, "ai.onnx.ml": 3})
    config.ONNX_PATH.write_bytes(onx.SerializeToString())
    X = pd.read_csv("dataset/processed/test.csv")[FEATURE_NAMES].values[:5000]
    a = model.predict_proba(X.astype(np.float32))[:, 1]
    b = OnnxModel(config.ONNX_PATH).predict_proba(X)[:, 1]
    r = {"samples": int(len(X)), "max_abs_probability_diff": float(np.abs(a - b).max()),
         "label_agreement": float(((a >= .5) == (b >= .5)).mean()), "onnx_bytes": config.ONNX_PATH.stat().st_size,
         "note": "Local Intel CPU comparison only; ONNX Runtime CPU provider."}
    (config.REPORTS_DIR / "onnx_comparison.json").write_text(json.dumps(r, indent=2))
    print(r)


if __name__ == "__main__":
    main()
