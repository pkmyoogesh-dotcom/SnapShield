import pytest
from backend.services.feature_extractor import FEATURE_NAMES, extract_features


def test_all_features_present_and_numeric():
    f = extract_features("https://example.com/a?b=1")
    assert list(f) == FEATURE_NAMES and all(isinstance(v, float) for v in f.values())

def test_https_and_ip():
    assert extract_features("https://example.com")["is_https"] == 1
    f = extract_features("http://192.168.10.5/login")
    assert f["is_https"] == 0 and f["has_ip_host"] == 1 and f["keyword_count"] >= 1

def test_special_characters():
    f = extract_features("http://user@evil.example.com/%2e%2e//x?a=1&b=2")
    assert f["has_at_symbol"] == 1 and f["num_percent_encoded"] == 2
    assert f["has_double_slash_in_path"] == 1 and f["num_ampersands"] == 1

def test_subdomains_and_keywords():
    f = extract_features("http://secure.login.verify.example.co/account")
    assert f["num_subdomains"] == 3 and f["keyword_count"] >= 4

def test_long_url():
    assert extract_features("https://example.com/" + "a" * 1500)["url_length"] > 1500
