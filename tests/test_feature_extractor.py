# tests/test_feature_extractor.py
"""Unit tests for the feature_extractor module."""
import os
import pytest
import joblib

from feature_extractor import extract_features

# Helper to load the saved feature order (if model already trained)
def load_feature_names():
    path = os.path.join("models", "feature_names.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    # Fallback list based on the extractor's URL features (HTML features are added later)
    from feature_extractor import url_features
    url_feats = list(url_features("https://example.com").keys())
    # Append HTML feature names (same as in feature_extractor.html_features)
    html_feat_names = [
        "num_links",
        "num_external_links",
        "num_internal_links",
        "external_link_ratio",
        "num_forms",
        "has_login_form",
        "num_password_fields",
        "num_iframes",
        "num_scripts",
        "has_onmouseover",
        "has_popup",
        "domain_in_title",
        "has_favicon",
        "num_images",
        "num_external_css",
        "suspicious_word_count_html",
    ]
    return url_feats + html_feat_names

@pytest.fixture(scope="module")
def expected_features():
    return load_feature_names()

def test_valid_https(expected_features):
    feats = extract_features("https://example.com")
    # All expected keys should be present
    for k in expected_features:
        assert k in feats
    assert feats["is_https"] == 1

def test_ip_address(expected_features):
    feats = extract_features("http://192.168.0.1/login")
    assert feats["has_ip"] == 1

def test_at_symbol(expected_features):
    feats = extract_features("http://user@example.com")
    assert feats["num_at"] >= 1

def test_many_subdomains(expected_features):
    feats = extract_features("http://a.b.c.d.example.com")
    assert feats["num_subdomains"] >= 3

def test_invalid_url():
    with pytest.raises(ValueError):
        extract_features("not a url")

def test_empty_string():
    with pytest.raises(ValueError):
        extract_features("")

def test_query_parameters(expected_features):
    feats = extract_features("https://example.com/search?q=test&lang=en")
    assert feats["num_query_params"] >= 2
