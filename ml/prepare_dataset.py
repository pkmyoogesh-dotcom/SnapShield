"""Prepare the PhiUSIIL Phishing URL dataset (UCI ML Repository, id 967).

Setup (manual download; needs internet):
  1. Open https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
  2. Download and unzip; place PhiUSIIL_Phishing_URL_Dataset.csv in dataset/raw/
  3. Confirm the license shown on that page (listed as CC BY 4.0) and cite the authors.
Original labels: 1 = legitimate, 0 = phishing. We convert to is_phishing (1 = phishing).
Only the raw URL string is used; the dataset's pre-computed page-content columns are ignored.

Leakage control: exact-duplicate URLs are dropped and the split is grouped by hostname,
so the same host never appears in more than one of train/val/test.
Run: python -m ml.prepare_dataset
"""
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from ml.feature_engineering import build_feature_frame

RAW = Path("dataset/raw/PhiUSIIL_Phishing_URL_Dataset.csv")
OUT = Path("dataset/processed")
SEED = 42


def main():
    if not RAW.exists():
        sys.exit(f"Missing {RAW}. See the setup steps at the top of ml/prepare_dataset.py.")
    df = pd.read_csv(RAW, usecols=["URL", "label"]).dropna()
    n_raw = len(df)
    df = df.assign(URL=df["URL"].astype(str).str.strip()).drop_duplicates("URL")
    n_dedup = len(df)
    df["is_phishing"] = 1 - df["label"].astype(int)
    feats, keep = build_feature_frame(df["URL"].tolist())
    df = df.iloc[keep].reset_index(drop=True)
    feats["is_phishing"] = df["is_phishing"].values
    groups = df["URL"].map(lambda u: (urlsplit(u).hostname or "").lower()).values

    gss = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=SEED)
    tr, rest = next(gss.split(feats, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=SEED)
    va, te = next(gss2.split(feats.iloc[rest], groups=groups[rest]))
    va, te = rest[va], rest[te]

    OUT.mkdir(parents=True, exist_ok=True)
    info = {"source": "PhiUSIIL Phishing URL Dataset, UCI ML Repository id 967",
            "license": "CC BY 4.0 (as listed on the UCI page; verify)",
            "rows_raw": n_raw, "rows_after_dedup": n_dedup, "rows_used": len(feats),
            "dropped_invalid_urls": n_dedup - len(feats), "split": "70/15/15 grouped by hostname", "seed": SEED}
    for name, idx in (("train", tr), ("val", va), ("test", te)):
        part = feats.iloc[idx]
        part.to_csv(OUT / f"{name}.csv", index=False)
        # keep raw URLs of the test split for evaluation/demo (from the real dataset)
        df.iloc[idx][["URL", "is_phishing"]].to_csv(OUT / f"{name}_urls.csv", index=False)
        info[f"{name}_rows"] = len(part)
        info[f"{name}_phishing_rate"] = round(float(part["is_phishing"].mean()), 4)
    (OUT / "dataset_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
