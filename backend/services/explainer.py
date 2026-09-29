"""Transparent, model-agnostic explanations.

Method (single-feature ablation vs. a safe baseline): each feature is replaced by the
median value seen among LEGITIMATE training URLs, the same model is re-run, and the
drop in phishing probability is recorded. Features whose reset lowers the score are
the ones pushing this URL toward phishing. Reason text is a template filled with the
URL's real feature values. Limitation: correlated features (e.g. url_length and
path_length) share credit non-additively.
"""
from __future__ import annotations

import numpy as np

from backend.services.feature_extractor import FEATURE_NAMES

LABELS = {
    "url_length": lambda v: f"Unusually long URL ({int(v)} characters)",
    "hostname_length": lambda v: f"Long hostname ({int(v)} characters)",
    "path_length": lambda v: f"Long URL path ({int(v)} characters)",
    "query_length": lambda v: f"Long query string ({int(v)} characters)",
    "num_subdomains": lambda v: f"Multiple subdomain levels ({int(v)})",
    "has_ip_host": lambda v: "Hostname is a raw IP address",
    "has_at_symbol": lambda v: "Contains '@', which can hide the real destination",
    "keyword_count": lambda v: f"Suspicious keyword(s) such as login/verify/secure detected ({int(v)})",
    "is_https": lambda v: "Connection is not HTTPS",
    "hostname_entropy": lambda v: f"Random-looking hostname (entropy {v:.2f} bits/char)",
    "path_entropy": lambda v: f"High path complexity (entropy {v:.2f} bits/char)",
    "num_special_chars": lambda v: f"Many special characters ({int(v)})",
    "num_percent_encoded": lambda v: f"URL-encoded characters present ({int(v)})",
    "risky_tld": lambda v: "Top-level domain frequently abused in phishing",
    "has_punycode": lambda v: "Punycode (xn--) hostname, possible look-alike domain",
    "hostname_hyphens": lambda v: f"Hyphens in hostname ({int(v)})",
    "has_double_slash_in_path": lambda v: "Double slash inside path",
}


def explain(vec, bundle, top_k: int = 5, min_delta: float = 0.02):
    n = len(FEATURE_NAMES)
    base = np.array([bundle.baseline[k] for k in FEATURE_NAMES], dtype=float)
    X = np.tile(np.asarray(vec, dtype=float), (n + 1, 1))
    for i in range(n):
        X[i + 1, i] = base[i]
    p = bundle.predict_proba(X)
    deltas = p[0] - p[1:]
    reasons, contributions = [], []
    for i in np.argsort(-deltas)[:top_k]:
        if deltas[i] < min_delta or vec[i] == base[i]:
            continue
        name, v = FEATURE_NAMES[i], float(vec[i])
        contributions.append({"feature": name, "value": v, "safe_baseline": float(base[i]),
                              "risk_contribution": round(float(deltas[i]), 4)})
        fmt = LABELS.get(name)
        reasons.append(fmt(v) if fmt else f"{name.replace('_', ' ')} = {v:g} (typical safe URL: {base[i]:g})")
    return reasons, contributions
