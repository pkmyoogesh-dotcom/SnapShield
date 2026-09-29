import pytest
from backend.services.feature_extractor import InvalidURLError, validate_url


@pytest.mark.parametrize("bad", ["", "   ", "not a url", "ftp://x.com", "javascript:alert(1)",
                                 "http://", "https://exa mple.com", "http://x.com:99999",
                                 "https://example.com/" + "a" * 2100, "file:///etc/passwd"])
def test_invalid(bad):
    with pytest.raises(InvalidURLError):
        validate_url(bad)

def test_valid_strips_whitespace():
    assert validate_url("  https://example.com ") == "https://example.com"
