# feature_extractor.py
"""
Utility functions that accept a raw URL string and return a feature dictionary
compatible with the trained Random Forest model.
"""

import re
from urllib.parse import urlparse, parse_qs
import ipaddress
import requests
from bs4 import BeautifulSoup
import tldextract

# ----------------------------------------------------------------------
# Constants (kept in sync with train_model.py)
# ----------------------------------------------------------------------
SUSPICIOUS_WORDS = [
    "login", "bank", "account", "verify", "update", "free", "lucky", "winner",
    "award", "gift", "confirm", "security", "ebay", "paypal", "password",
    "signin", "admin", "webscr", "pay", "checkout",
]

SHORTENING_SERVICES = [
    "bit.ly", "goo.gl", "t.co", "tinyurl.com", "ow.ly", "bit.do",
    "buff.ly", "adf.ly", "cutt.ly", "is.gd", "shorte.st",
]

MAX_HTML_BYTES = 500_000   # 500 KB
REQUEST_TIMEOUT = 5        # seconds
MAX_REDIRECTS = 3
USER_AGENT = "PhishDetect/1.0 (+https://github.com/yourname)"

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------

def is_ip_address(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname)
        return True
    except Exception:
        return False


def count_subdomains(hostname: str) -> int:
    ext = tldextract.extract(hostname)
    return len(ext.subdomain.split('.')) if ext.subdomain else 0

# ----------------------------------------------------------------------
# URL‑based feature extraction (mirrored from train_model.py)
# ----------------------------------------------------------------------

def url_features(url: str) -> dict:
    """Return a dict of URL‑only features.
    The implementation mirrors `train_model.py` so that training and live
    prediction use identical calculations.
    """
    # Normalise – add scheme if missing for parsing only
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    length = len(url)
    hostname_len = len(hostname)
    path_len = len(path)
    query_len = len(query)
    # Simple numeric counters
    feats = {
        "url_length": length,
        "hostname_length": hostname_len,
        "path_length": path_len,
        "query_length": query_len,
        "num_dots": url.count('.'),
        "num_hyphens": url.count('-'),
        "num_at": url.count('@'),
        "num_question_marks": url.count('?'),
        "num_ampersands": url.count('&'),
        "num_equals": url.count('='),
        "num_underscores": url.count('_'),
        "num_percents": url.count('%'),
        "num_slashes": url.count('/'),
        "num_colons": url.count(':'),
        "num_commas": url.count(','),
        "digit_ratio": sum(c.isdigit() for c in url) / length if length else 0,
        "digit_ratio_hostname": sum(c.isdigit() for c in hostname) / hostname_len if hostname_len else 0,
        "num_subdomains": count_subdomains(hostname),
        "has_ip": int(is_ip_address(hostname)),
        "is_https": int(parsed.scheme.lower() == 'https'),
        "http_in_path": int('http' in path.lower()),
        "is_shortening": int(any(svc in hostname.lower() for svc in SHORTENING_SERVICES)),
        "suspicious_tld": int(
            ('.' + hostname.split('.')[-1].lower()) in {'.tk', '.ml', '.ga', '.cf', '.gq'}
        ),
        "has_port": int(parsed.port is not None),
        "punycode": int(hostname.startswith('xn--')),
        "brand_in_domain": int(any(b in hostname.lower() for b in ['paypal','apple','google','amazon','microsoft','bank'])),
        "brand_in_path": int(any(b in path.lower() for b in ['paypal','apple','google','amazon','microsoft','bank'])),
        "suspicious_word_count": sum(url.lower().count(w) for w in SUSPICIOUS_WORDS),
        "num_query_params": len(parse_qs(query)),
    }
    return feats

# ----------------------------------------------------------------------
# Safe HTML retrieval (no redirects, size limit, timeout)
# ----------------------------------------------------------------------

def safe_fetch_html(url: str):
    """Return the raw HTML string if the request succeeds safely.
    Returns *None* on any failure (timeout, non‑HTML content, too many redirects, etc.).
    """
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
            stream=True,
        )
        if len(resp.history) > MAX_REDIRECTS:
            return None
        ct = resp.headers.get('Content-Type', '')
        if 'html' not in ct.lower():
            return None
        content = resp.raw.read(MAX_HTML_BYTES, decode_content=True)
        return content.decode(errors='ignore')
    except Exception:
        return None

# ----------------------------------------------------------------------
# HTML‑based feature extraction (mirrored from train_model.py)
# ----------------------------------------------------------------------

def html_features(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, 'html.parser')
    # Links
    links = soup.find_all('a', href=True)
    num_links = len(links)
    external_links = sum(
        1 for a in links
        if urlparse(a['href']).netloc and urlparse(a['href']).netloc != urlparse(url).netloc
    )
    internal_links = num_links - external_links
    external_ratio = external_links / num_links if num_links else 0
    # Forms
    forms = soup.find_all('form')
    num_forms = len(forms)
    login_form = any(
        any(inp.get('type') == 'password' for inp in form.find_all('input'))
        for form in forms
    )
    num_password_fields = sum(1 for inp in soup.find_all('input', {'type': 'password'}))
    # Iframes, scripts, etc.
    num_iframes = len(soup.find_all('iframe'))
    num_scripts = len(soup.find_all('script'))
    has_onmouseover = int(bool(soup.select('[onmouseover]')))
    has_popup = int(bool(soup.select("[onclick*='window.open']")))
    # Title vs domain
    title = soup.title.string if soup.title else ''
    domain = urlparse(url).hostname or ''
    domain_in_title = int(domain.lower() in title.lower())
    # Favicon, images, external CSS
    has_favicon = int(bool(soup.find('link', rel=lambda x: x and 'icon' in x)))
    num_images = len(soup.find_all('img'))
    external_css = len([
        link for link in soup.find_all('link', rel='stylesheet')
        if link.get('href') and urlparse(link['href']).netloc
    ])
    # Suspicious words in page text
    page_text = soup.get_text(separator=' ').lower()
    suspicious_word_cnt = sum(page_text.count(w) for w in SUSPICIOUS_WORDS)
    return {
        "num_links": num_links,
        "num_external_links": external_links,
        "num_internal_links": internal_links,
        "external_link_ratio": external_ratio,
        "num_forms": num_forms,
        "has_login_form": int(login_form),
        "num_password_fields": num_password_fields,
        "num_iframes": num_iframes,
        "num_scripts": num_scripts,
        "has_onmouseover": has_onmouseover,
        "has_popup": has_popup,
        "domain_in_title": domain_in_title,
        "has_favicon": has_favicon,
        "num_images": num_images,
        "num_external_css": external_css,
        "suspicious_word_count_html": suspicious_word_cnt,
    }

# ----------------------------------------------------------------------
# Public API used by the Streamlit app and the unit tests
# ----------------------------------------------------------------------

def extract_features(url: str) -> dict:
    """Return a feature dict that matches the training‑time column order.
    The dict also contains a non‑model field ``html_status`` indicating whether
    HTML analysis succeeded ("available") or fell back to neutral defaults
    ("unavailable").
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a non‑empty string")
    url = url.strip()
    url_feats = url_features(url)
    html = safe_fetch_html(url)
    if html:
        html_feats = html_features(html, url)
        html_status = 'available'
    else:
        # Neutral fallback values for all HTML features
        html_feats = {
            "num_links": 0,
            "num_external_links": 0,
            "num_internal_links": 0,
            "external_link_ratio": 0.0,
            "num_forms": 0,
            "has_login_form": 0,
            "num_password_fields": 0,
            "num_iframes": 0,
            "num_scripts": 0,
            "has_onmouseover": 0,
            "has_popup": 0,
            "domain_in_title": 0,
            "has_favicon": 0,
            "num_images": 0,
            "num_external_css": 0,
            "suspicious_word_count_html": 0,
        }
        html_status = 'unavailable'
    features = {**url_feats, **html_feats}
    features['html_status'] = html_status
    return features
