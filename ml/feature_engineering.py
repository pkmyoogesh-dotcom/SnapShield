"""Builds feature tables using the SAME extractor the API uses (no train/serve skew)."""
import pandas as pd

from backend.services.feature_extractor import FEATURE_NAMES, extract_features, validate_url, InvalidURLError

__all__ = ["FEATURE_NAMES", "build_feature_frame"]


def build_feature_frame(urls):
    rows, keep = [], []
    for i, u in enumerate(urls):
        try:
            rows.append(extract_features(u))
            keep.append(i)
        except InvalidURLError:
            pass
    return pd.DataFrame(rows, columns=FEATURE_NAMES), keep
