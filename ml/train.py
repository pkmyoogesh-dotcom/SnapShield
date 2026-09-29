"""Train candidate models, select by VALIDATION F1, save model + metadata.
Run: python -m ml.train"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backend import config
from backend.services.feature_extractor import FEATURE_NAMES

P = Path("dataset/processed")


def main():
    tr, va = pd.read_csv(P / "train.csv"), pd.read_csv(P / "val.csv")
    Xtr, ytr, Xva, yva = tr[FEATURE_NAMES].values, tr["is_phishing"].values, va[FEATURE_NAMES].values, va["is_phishing"].values
    candidates = {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, class_weight="balanced", random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42),
    }
    results, best_name, best = {}, None, -1
    for name, m in candidates.items():
        m.fit(Xtr, ytr)
        proba = m.predict_proba(Xva)[:, 1]
        f1 = f1_score(yva, proba >= 0.5)
        results[name] = {"val_f1": round(f1, 5), "val_roc_auc": round(roc_auc_score(yva, proba), 5)}
        print(name, results[name])
        if f1 > best:
            best_name, best = name, f1
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidates[best_name], config.MODEL_PATH)
    baseline = tr[tr["is_phishing"] == 0][FEATURE_NAMES].median().to_dict()
    meta = {"model_type": best_name, "selected_by": "validation F1 @ 0.5",
            "candidate_validation_results": results, "feature_names": FEATURE_NAMES,
            "class_labels": {"0": "legitimate", "1": "phishing"},
            "trained_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": json.loads((P / "dataset_info.json").read_text()),
            "preprocessing": "lexical URL features (backend/services/feature_extractor.py); scaling inside pipeline for LogisticRegression only",
            "safe_baseline": baseline,
            "thresholds": {"suspicious": config.SUSPICIOUS_THRESHOLD, "phishing": config.PHISHING_THRESHOLD}}
    (config.MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print("Selected:", best_name)


if __name__ == "__main__":
    main()
