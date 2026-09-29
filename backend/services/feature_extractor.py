"""Lexical URL feature extraction. Pure string analysis: the URL is never fetched."""
from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urlsplit

MAX_URL_LENGTH = 2048
KEYWORDS = ("login", "signin", "sign-in", "verify", "account", "update", "secure",
            "banking", "confirm", "password", "paypal", "webscr", "wallet",
            "suspend", "invoice", "billing", "auth")
RISKY_TLDS = {"zip", "mov", "xyz", "top", "tk", "ml", "ga", "cf", "gq", "click",
              "country", "work", "rest", "icu", "buzz", "cyou", "monster"}

FEATURE_NAMES = [
    "url_length", "hostname_length", "path_length", "query_length", "num_dots",
    "num_hyphens", "num_underscores", "num_slashes", "num_question_marks",
    "num_equals", "num_ampersands", "num_special_chars", "num_digits", "digit_ratio",
    "num_subdomains", "is_https", "has_ip_host", "keyword_count", "has_at_symbol",
    "num_percent_encoded", "hostname_entropy", "path_entropy", "path_depth",
    "hostname_hyphens", "host_digit_count", "risky_tld", "has_port", "has_punycode",
    "longest_host_label", "has_double_slash_in_path",
]


class InvalidURLError(ValueError):
    """Raised when the submitted string is not an acceptable http(s) URL."""


def validate_url(raw: str) -> str:
    if not isinstance(raw, str):
        raise InvalidURLError("URL must be a string.")
    url = raw.strip()
    if not url:
        raise InvalidURLError("URL is empty.")
    if len(url) > MAX_URL_LENGTH:
        raise InvalidURLError(f"URL exceeds {MAX_URL_LENGTH} characters.")
    if re.search(r"[\s\x00-\x1f\x7f]", url):
        raise InvalidURLError("URL contains whitespace or control characters.")
    try:
        parts = urlsplit(url)
        host = parts.hostname
        _ = parts.port
    except ValueError:
        raise InvalidURLError("URL could not be parsed.") from None
    if parts.scheme.lower() not in ("http", "https"):
        raise InvalidURLError("Only http:// and https:// URLs are supported.")
    if not host:
        raise InvalidURLError("URL has no hostname.")
    return url


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    n = len(text)
    return -sum(c / n * math.log2(c / n) for c in Counter(text).values())


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def extract_features(url: str) -> dict[str, float]:
    url = validate_url(url)
    p = urlsplit(url)
    host = (p.hostname or "").lower()
    path, query = p.path or "", p.query or ""
    is_ip = _is_ip(host)
    labels = host.split(".")
    low = url.lower()
    try:
        has_port = p.port is not None
    except ValueError:
        has_port = False
    n_digits = sum(ch.isdigit() for ch in url)
    f = {
        "url_length": len(url),
        "hostname_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_question_marks": url.count("?"),
        "num_equals": url.count("="),
        "num_ampersands": url.count("&"),
        "num_special_chars": sum(not ch.isalnum() for ch in url),
        "num_digits": n_digits,
        "digit_ratio": n_digits / len(url),
        "num_subdomains": 0 if is_ip else max(len(labels) - 2, 0),
        "is_https": int(p.scheme.lower() == "https"),
        "has_ip_host": int(is_ip),
        "keyword_count": sum(low.count(k) for k in KEYWORDS),
        "has_at_symbol": int("@" in url),
        "num_percent_encoded": len(re.findall(r"%[0-9a-fA-F]{2}", url)),
        "hostname_entropy": shannon_entropy(host),
        "path_entropy": shannon_entropy(path),
        "path_depth": len([s for s in path.split("/") if s]),
        "hostname_hyphens": host.count("-"),
        "host_digit_count": sum(ch.isdigit() for ch in host),
        "risky_tld": int((not is_ip) and labels[-1] in RISKY_TLDS),
        "has_port": int(has_port),
        "has_punycode": int("xn--" in host),
        "longest_host_label": max(len(x) for x in labels),
        "has_double_slash_in_path": int("//" in path),
    }
    return {k: float(f[k]) for k in FEATURE_NAMES}


def feature_vector(features: dict[str, float]) -> list[float]:
    return [features[n] for n in FEATURE_NAMES]
