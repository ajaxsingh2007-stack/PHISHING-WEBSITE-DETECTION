# explanation.py
"""
Generate a short, user‑friendly explanation from the extracted feature
values and the Random Forest model’s global feature importance.
"""
import joblib
import numpy as np
from typing import Dict, List

# Paths relative to project root
MODEL_PATH = "models/phishing_model.pkl"
FEATURES_PATH = "models/feature_names.pkl"

# Load once (cached when imported)
clf = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)

# Global importance sorted descending
global_importance = dict(
    sorted(
        zip(feature_names, clf.feature_importances_),
        key=lambda kv: kv[1],
        reverse=True,
    )
)

# Simple thresholds for what we consider a "suspicious" value.
# These were chosen heuristically based on the training data distribution.
SUSPICIOUS_THRESHOLDS = {
    "url_length": 75,
    "hostname_length": 30,
    "path_length": 50,
    "num_dots": 3,
    "num_hyphens": 2,
    "num_at": 0,
    "num_question_marks": 1,
    "num_ampersands": 2,
    "num_equals": 2,
    "num_underscores": 2,
    "num_percents": 2,
    "num_subdomains": 2,
    "has_ip": 1,
    "is_https": 0,  # not HTTPS is suspicious
    "http_in_path": 1,
    "is_shortening": 1,
    "suspicious_tld": 1,
    "punycode": 1,
    "brand_in_domain": 1,
    "brand_in_path": 1,
    "suspicious_word_count": 1,
    "num_query_params": 3,
    # HTML thresholds
    "external_link_ratio": 0.6,
    "has_login_form": 1,
    "num_iframes": 2,
    "has_onmouseover": 1,
    "has_popup": 1,
    "domain_in_title": 0,
    "suspicious_word_count_html": 2,
}

def _flag_suspicious(feats: Dict[str, float]) -> List[str]:
    """Return a list of feature names that exceed their suspicious thresholds."""
    flagged = []
    for name, thresh in SUSPICIOUS_THRESHOLDS.items():
        if name not in feats:
            continue
        val = feats[name]
        # Binary features (0/1) – a value of 1 means the condition is present.
        if isinstance(val, (int, np.integer)):
            if val >= thresh and thresh > 0:
                flagged.append(name)
        else:  # float / ratio
            if val >= thresh:
                flagged.append(name)
    return flagged

# Human‑readable mapping of technical feature names
FRIENDLY = {
    "url_length": "the URL is unusually long",
    "hostname_length": "the hostname is long",
    "path_length": "the path part of the URL is long",
    "num_dots": "many subdomains (multiple dots)",
    "num_hyphens": "several hyphens in the URL",
    "num_at": "`@` symbol appears in the URL",
    "num_question_marks": "many query markers (`?`)",
    "num_ampersands": "multiple ampersands (`&`)",
    "num_equals": "many equals signs (`=`)",
    "num_underscores": "underscores in the URL",
    "num_percents": "percent symbols (`%`)",
    "num_subdomains": "deep sub‑domain hierarchy",
    "has_ip": "the host part is an IP address",
    "is_https": "the site does **not** use HTTPS",
    "http_in_path": "`http` appears inside the URL path",
    "is_shortening": "the link uses a known URL‑shortening service",
    "suspicious_tld": "the top‑level domain is commonly used for malicious sites",
    "punycode": "the domain uses Punycode (possible IDN spoofing)",
    "brand_in_domain": "a popular brand name appears in the domain",
    "brand_in_path": "a popular brand name appears in the path",
    "suspicious_word_count": "the URL contains phishing‑related words",
    "num_query_params": "many query parameters are present",
    "external_link_ratio": "most links on the page point to external sites",
    "has_login_form": "the page contains a login form",
    "num_iframes": "several iframes are embedded",
    "has_onmouseover": "`onmouseover` JavaScript handler detected",
    "has_popup": "popup‑triggering JavaScript detected",
    "domain_in_title": "the domain name does **not** appear in the page title",
    "suspicious_word_count_html": "suspicious words appear in the page text",
}

def generate_explanation(features: Dict[str, float], prob_phish: float) -> str:
    """Create a paragraph explaining the model decision.

    Parameters
    ----------
    features: dict
        Feature dictionary returned by ``feature_extractor.extract_features``.
    prob_phish: float
        Probability for the *phishing* class (0‑1).
    """
    suspicious = _flag_suspicious(features)
    # Sort by global importance – most influential first
    suspicious_sorted = sorted(suspicious, key=lambda x: global_importance.get(x, 0), reverse=True)

    explanation_lines = []
    for f in suspicious_sorted:
        explanation_lines.append(f"• {FRIENDLY.get(f, f.replace('_', ' '))}")

    # Reassuring indicators (simple list)
    reassuring = []
    if features.get('is_https', 0) == 1:
        reassuring.append("• the connection uses HTTPS")
    if features.get('domain_in_title', 0) == 1:
        reassuring.append("• the domain name appears in the page title")

    # Decision wording based on probability
    decision = "potential phishing" if prob_phish >= 0.5 else "likely legitimate"
    result = f"The model classified the URL as **{decision}** (phishing probability = {prob_phish:.2%})."
    if explanation_lines:
        result += "\n\nKey characteristics that contributed to this assessment:\n" + "\n".join(explanation_lines)
    else:
        result += " No strong suspicious indicators were detected."
    if reassuring:
        result += "\n\nReassuring signs:\n" + "\n".join(reassuring)
    result += "\n\n*Note: these observations are statistical cues and do not guarantee safety.*"
    return result
