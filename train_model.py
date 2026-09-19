# train_model.py
"""
Train a Random Forest model for phishing‑website detection.

Stage 1 – Inspect the dataset (prints a concise summary).
Stage 2 – Engineer reproducible URL‑based & safe HTML‑based features.
Stage 3 – Train / evaluate / save the model and reports.
"""

import os
import sys
import warnings
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ----------------------------------------------------------------------
# Helper: load CSV (graceful if missing)
# ----------------------------------------------------------------------
def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        print(f"[ERROR] Dataset not found at {csv_path!s}")
        sys.exit(1)
    return pd.read_csv(csv_path)

# ----------------------------------------------------------------------
# Stage 1 – Dataset inspection
# ----------------------------------------------------------------------
def inspect_dataset(df: pd.DataFrame):
    print("\n=== DATASET OVERVIEW ===")
    print(f"Rows, Columns      : {df.shape}")
    print(f"Column names       : {list(df.columns)}\n")
    print("Data types:")
    print(df.dtypes, "\n")
    print("Missing values per column:")
    print(df.isnull().sum(), "\n")
    dup = df.duplicated().sum()
    print(f"Duplicate rows    : {dup}\n")
    # Find target column automatically (common names)
    target_candidates = [c for c in df.columns if any(tok in c.lower() for tok in ["status", "label", "target", "class"])]
    if not target_candidates:
        # fallback to column 's' which holds the label in this dataset
        if 's' in df.columns:
            target_col = 's'
            print("[INFO] Falling back to column 's' as target.")
        else:
            print("[WARNING] Could not locate a target column.")
            return None
    else:
        target_col = target_candidates[0]
    print(f"Target column      : {target_col}")
    print("Class distribution :")
    print(df[target_col].value_counts(), "\n")
    return target_col

# ----------------------------------------------------------------------
# Feature engineering – only reproducible features
# ----------------------------------------------------------------------
from urllib.parse import urlparse, parse_qs
import re
import tldextract
import ipaddress
import requests
from bs4 import BeautifulSoup

SUSPICIOUS_WORDS = [
    "login", "bank", "account", "verify", "update", "free", "lucky", "winner",
    "award", "gift", "confirm", "security", "ebay", "paypal", "password",
    "signin", "admin", "webscr", "pay", "checkout",
]

SHORTENING_SERVICES = [
    "bit.ly", "goo.gl", "t.co", "tinyurl.com", "ow.ly", "bit.do",
    "buff.ly", "adf.ly", "cutt.ly", "is.gd", "shorte.st",
]

def is_ip_address(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname)
        return True
    except Exception:
        return False

def count_subdomains(hostname: str) -> int:
    ext = tldextract.extract(hostname)
    return len(ext.subdomain.split('.')) if ext.subdomain else 0

def url_features(url: str) -> dict:
    # Normalise – add scheme if missing
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
    num_dots = url.count('.')
    num_hyphens = url.count('-')
    num_at = url.count('@')
    num_q = url.count('?')
    num_amp = url.count('&')
    num_eq = url.count('=')
    num_underscore = url.count('_')
    num_percent = url.count('%')
    num_slash = url.count('/')
    num_colon = url.count(':')
    num_comma = url.count(',')
    digit_cnt = sum(c.isdigit() for c in url)
    digit_ratio = digit_cnt / length if length else 0
    digit_host_cnt = sum(c.isdigit() for c in hostname)
    digit_host_ratio = digit_host_cnt / hostname_len if hostname_len else 0
    subdomains = count_subdomains(hostname)
    has_ip = int(is_ip_address(hostname))
    is_https = int(parsed.scheme.lower() == 'https')
    http_in_path = int('http' in path.lower())
    is_short = int(any(svc in hostname.lower() for svc in SHORTENING_SERVICES))
    suspicious_tlds = {'.tk', '.ml', '.ga', '.cf', '.gq'}
    tld = '.' + hostname.split('.')[-1] if '.' in hostname else ''
    suspicious_tld = int(tld in suspicious_tlds)
    has_port = int(parsed.port is not None)
    punycode = int(hostname.startswith('xn--'))
    brand_names = ['paypal', 'apple', 'google', 'amazon', 'microsoft', 'bank']
    brand_in_domain = int(any(b in hostname.lower() for b in brand_names))
    brand_in_path = int(any(b in path.lower() for b in brand_names))
    word_cnt = sum(url.lower().count(w) for w in SUSPICIOUS_WORDS)
    query_params = len(parse_qs(query))
    return {
        'url_length': length,
        'hostname_length': hostname_len,
        'path_length': path_len,
        'query_length': query_len,
        'num_dots': num_dots,
        'num_hyphens': num_hyphens,
        'num_at': num_at,
        'num_question_marks': num_q,
        'num_ampersands': num_amp,
        'num_equals': num_eq,
        'num_underscores': num_underscore,
        'num_percents': num_percent,
        'num_slashes': num_slash,
        'num_colons': num_colon,
        'num_commas': num_comma,
        'digit_ratio': digit_ratio,
        'digit_ratio_hostname': digit_host_ratio,
        'num_subdomains': subdomains,
        'has_ip': has_ip,
        'is_https': is_https,
        'http_in_path': http_in_path,
        'is_shortening': is_short,
        'suspicious_tld': suspicious_tld,
        'has_port': has_port,
        'punycode': punycode,
        'brand_in_domain': brand_in_domain,
        'brand_in_path': brand_in_path,
        'suspicious_word_count': word_cnt,
        'num_query_params': query_params,
    }

# ----------------------------------------------------------------------
# HTML‑based feature extraction (safe, size‑limited)
# ----------------------------------------------------------------------
MAX_HTML_BYTES = 500_000   # 500 KB
REQUEST_TIMEOUT = 5        # seconds
MAX_REDIRECTS = 3
USER_AGENT = "PhishDetect/1.0 (+https://github.com/yourname)"

def safe_fetch_html(url: str):
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

def html_features(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, 'html.parser')
    links = soup.find_all('a', href=True)
    num_links = len(links)
    external_links = sum(
        1 for a in links
        if urlparse(a['href']).netloc and urlparse(a['href']).netloc != urlparse(url).netloc
    )
    internal_links = num_links - external_links
    external_ratio = external_links / num_links if num_links else 0
    forms = soup.find_all('form')
    num_forms = len(forms)
    login_form = any(
        any(inp.get('type') == 'password' for inp in form.find_all('input'))
        for form in forms
    )
    num_password_fields = sum(1 for inp in soup.find_all('input', {'type': 'password'}))
    num_iframes = len(soup.find_all('iframe'))
    num_scripts = len(soup.find_all('script'))
    has_onmouseover = int(bool(soup.select('[onmouseover]')))
    has_popup = int(bool(soup.select("[onclick*='window.open']")))
    title = soup.title.string if soup.title else ''
    domain = urlparse(url).hostname or ''
    domain_in_title = int(domain.lower() in title.lower())
    has_favicon = int(bool(soup.find('link', rel=lambda x: x and 'icon' in x)))
    num_images = len(soup.find_all('img'))
    external_css = len([
        link for link in soup.find_all('link', rel='stylesheet')
        if link.get('href') and urlparse(link['href']).netloc
    ])
    page_text = soup.get_text(separator=' ').lower()
    suspicious_word_cnt = sum(page_text.count(w) for w in SUSPICIOUS_WORDS)
    return {
        'num_links': num_links,
        'num_external_links': external_links,
        'num_internal_links': internal_links,
        'external_link_ratio': external_ratio,
        'num_forms': num_forms,
        'has_login_form': int(login_form),
        'num_password_fields': num_password_fields,
        'num_iframes': num_iframes,
        'num_scripts': num_scripts,
        'has_onmouseover': has_onmouseover,
        'has_popup': has_popup,
        'domain_in_title': domain_in_title,
        'has_favicon': has_favicon,
        'num_images': num_images,
        'num_external_css': external_css,
        'suspicious_word_count_html': suspicious_word_cnt,
    }

# ----------------------------------------------------------------------
# Combine URL + HTML features (fallback if HTML missing)
# ----------------------------------------------------------------------
def extract_features_row(row_url: str) -> dict:
    url_feats = url_features(row_url)
    html = safe_fetch_html(row_url)
    if html:
        html_feats = html_features(html, row_url)
        html_status = 'available'
    else:
        html_feats = {
            'num_links': 0,
            'num_external_links': 0,
            'num_internal_links': 0,
            'external_link_ratio': 0.0,
            'num_forms': 0,
            'has_login_form': 0,
            'num_password_fields': 0,
            'num_iframes': 0,
            'num_scripts': 0,
            'has_onmouseover': 0,
            'has_popup': 0,
            'domain_in_title': 0,
            'has_favicon': 0,
            'num_images': 0,
            'num_external_css': 0,
            'suspicious_word_count_html': 0,
        }
        html_status = 'unavailable'
    feats = {**url_feats, **html_feats}
    feats['html_status'] = html_status
    return feats

# ----------------------------------------------------------------------
# Main training pipeline
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Train phishing detector')
    parser.add_argument('--inspect-only', action='store_true', help='Only inspect the dataset and exit.')
    args = parser.parse_args()

    csv_path = Path('phishing.csv')
    df = load_data(csv_path)
    target_col = inspect_dataset(df)
    if target_col is None:
        sys.exit(1)

    # Extract URL‑based features for the whole dataset (no HTML download here)
    print('\nExtracting URL‑based features from the dataset...')
    url_feat_df = pd.DataFrame([url_features(u) for u in df['url'].astype(str)], index=df.index)
    data = pd.concat([url_feat_df, df[target_col]], axis=1)

    # Encode target
    label_map = {'legitimate': 0, 'phishing': 1}
    data['label'] = data[target_col].map(label_map)
    if data['label'].isnull().any():
        print('[ERROR] Unknown label values encountered.')
        sys.exit(1)

    X = data.drop(columns=[target_col, 'label'])
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced',
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print('\n=== EVALUATION ===')
    print(f'Accuracy : {acc:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall   : {rec:.4f}')
    print(f'F1‑score : {f1:.4f}')

    # Save classification report
    report = classification_report(y_test, y_pred, target_names=['legitimate', 'phishing'])
    Path('reports/classification_report.txt').write_text(report)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legit', 'Phish'], yticklabels=['Legit', 'Phish'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('reports/confusion_matrix.png')
    plt.close()

    # Feature importance plot (top 20)
    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances[:20], y=importances.index[:20])
    plt.title('Top 20 Feature Importances')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('reports/feature_importance.png')
    plt.close()

    # Save model and feature order
    Path('models').mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, 'models/phishing_model.pkl')
    joblib.dump(list(X.columns), 'models/feature_names.pkl')

    print('\nModel, feature list, and reports have been saved under `models/` and `reports/`.')
    print('Run `streamlit run app.py` to launch the web UI.')

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
