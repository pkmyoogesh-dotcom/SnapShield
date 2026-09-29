"""LOCAL INTEL DEVELOPMENT BENCHMARK (not Snapdragon). Run: python -m ml.benchmark"""
import json
import platform
import statistics
import time

import numpy as np
import pandas as pd

from backend import config
from backend.services.feature_extractor import FEATURE_NAMES, extract_features, feature_vector
from backend.services.predictor import ModelBundle


def timeit(fn, xs):
    ts = []
    for x in xs:
        t = time.perf_counter(); fn(x); ts.append((time.perf_counter() - t) * 1000)
    return {"mean_ms": statistics.fmean(ts), "median_ms": statistics.median(ts)}


def main():
    urls = pd.read_csv("dataset/processed/test_urls.csv")["URL"].head(1000).tolist()
    out = {"label": "LOCAL INTEL DEVELOPMENT BENCHMARK - NOT Snapdragon performance",
           "machine": platform.processor() or platform.machine(), "python": platform.python_version(),
           "test_samples": len(urls)}
    t = time.perf_counter(); b = ModelBundle.load(); out["sklearn_load_ms"] = (time.perf_counter() - t) * 1000
    out["sklearn_model_bytes"] = config.MODEL_PATH.stat().st_size
    vecs = [feature_vector(extract_features(u)) for u in urls]
    out["feature_extraction"] = timeit(extract_features, urls)
    out["sklearn_single_inference"] = timeit(lambda v: b.predict_proba([v]), vecs)
    if config.ONNX_PATH.exists():
        from backend.services.onnx_predictor import OnnxModel
        t = time.perf_counter(); o = OnnxModel(config.ONNX_PATH); out["onnx_load_ms"] = (time.perf_counter() - t) * 1000
        out["onnx_model_bytes"] = config.ONNX_PATH.stat().st_size
        out["onnx_single_inference"] = timeit(lambda v: o.predict_proba([v]), vecs)
    (config.REPORTS_DIR / "local_benchmark.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
