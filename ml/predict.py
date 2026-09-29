"""CLI: python -m ml.predict https://example.com"""
import json
import sys

from backend.services.explainer import explain
from backend.services.feature_extractor import extract_features, feature_vector
from backend.services.predictor import ModelBundle, classify

if __name__ == "__main__":
    b = ModelBundle.load()
    f = extract_features(sys.argv[1])
    v = feature_vector(f)
    s = float(b.predict_proba([v])[0])
    print(json.dumps({"classification": classify(s), "risk_score": round(s, 4), "reasons": explain(v, b)[0]}, indent=2))
