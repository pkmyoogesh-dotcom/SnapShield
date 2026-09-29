"""Evaluate on the held-out TEST split. Run: python -m ml.evaluate"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from backend import config
from backend.services.feature_extractor import FEATURE_NAMES


def main():
    te = pd.read_csv("dataset/processed/test.csv")
    y = te["is_phishing"].values
    model = joblib.load(config.MODEL_PATH)
    p = model.predict_proba(te[FEATURE_NAMES].values)[:, 1]
    pred = (p >= 0.5).astype(int)
    cm = confusion_matrix(y, pred)
    three = np.where(p >= config.PHISHING_THRESHOLD, "PHISHING", np.where(p >= config.SUSPICIOUS_THRESHOLD, "SUSPICIOUS", "SAFE"))
    m = {"split": "test", "n_samples": int(len(y)), "decision_threshold_binary": 0.5,
         "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
         "recall": recall_score(y, pred), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, p),
         "confusion_matrix": {"labels": ["legitimate", "phishing"], "matrix": cm.tolist()},
         "three_class_thresholds": {"suspicious": config.SUSPICIOUS_THRESHOLD, "phishing": config.PHISHING_THRESHOLD},
         "three_class_counts_by_true_label": {
             lab: {c: int(((y == v) & (three == c)).sum()) for c in ("SAFE", "SUSPICIOUS", "PHISHING")}
             for lab, v in (("legitimate", 0), ("phishing", 1))}}
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "model_metrics.json").write_text(json.dumps(m, indent=2))
    txt = "\n".join(f"{k}: {v}" for k, v in m.items())
    (config.REPORTS_DIR / "model_evaluation.txt").write_text(txt)
    print(txt)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, v, ha="center", va="center")
        ax.set_xticks([0, 1], ["legit", "phish"]); ax.set_yticks([0, 1], ["legit", "phish"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        fig.tight_layout(); fig.savefig(config.REPORTS_DIR / "confusion_matrix.png", dpi=120)
    except Exception as e:  # plotting is optional
        print("confusion_matrix.png skipped:", e)


if __name__ == "__main__":
    main()
