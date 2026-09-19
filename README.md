# README.md

# Explainable Phishing Website Detection Using Random Forest Classification

## Project Objective

This project builds a **beginner‑friendly** machine‑learning pipeline that detects phishing websites. Users can paste any URL into a simple Streamlit web interface and receive:
1. A prediction (Likely Legitimate | Potential Phishing | Unable to Analyze).
2. The model‑estimated probability.
3. An explanation that points out which URL/HTML characteristics contributed to the decision.
4. The actual feature values used for the prediction.
5. A clear disclaimer that the result is only an estimate.

The model is trained on the **Web Page Phishing Detection Dataset** from Kaggle (`phishing.csv`). Only features that can be **re‑computed at prediction time** (URL‑based and safe HTML‑based) are used, preventing data leakage.

---

## Technology Stack

- **Python** 3.9+
- **pandas**, **NumPy** – data handling
- **scikit‑learn** – Random Forest classifier
- **joblib** – model persistence
- **Requests** – safe HTTP fetching
- **BeautifulSoup4** – HTML parsing
- **Streamlit** – lightweight web UI
- **Matplotlib** & **Seaborn** – evaluation plots

---

## Project Structure

```
PhishingWebsiteDetector/
├── phishing.csv                # Kaggle dataset (place here)
├── app.py                     # Streamlit front‑end
├── train_model.py             # Data inspection, feature engineering, training, evaluation
├── feature_extractor.py        # Reproducible URL & HTML feature extraction
├── explanation.py             # Human‑readable explanation generator
├── requirements.txt           # Python dependencies
├── README.md                  # This document
├── models/
│   ├── phishing_model.pkl     # Trained RandomForest model
│   └── feature_names.pkl      # Ordered list of feature names used by the model
├── reports/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   └── feature_importance.png
└── tests/
    └── test_feature_extractor.py
```

---

## Setup Instructions

1. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # macOS / Linux
   # .\\venv\\Scripts\\activate   # Windows
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Place the dataset**
   - Download the *Web Page Phishing Detection Dataset* from Kaggle.
   - Rename the CSV to **`phishing.csv`** and copy it into the project root (`PhishingWebsiteDetector/`).
4. **Train the model**
   ```bash
   python train_model.py
   ```
   This will:
   - Print dataset statistics.
   - Engineer reproducible features.
   - Train a Random Forest classifier.
   - Save the model and feature list to `models/`.
   - Generate evaluation plots in `reports/`.
5. **Run the web app**
   ```bash
   streamlit run app.py
   ```
   Open the displayed URL in your browser (usually `http://localhost:8501`).

---

## How the Feature Extractor Works

`feature_extractor.py` contains two main parts:
1. **URL‑based features** – length, number of dots, presence of IP address, HTTPS usage, suspicious words, etc.
2. **HTML‑based features** – number of links, external/internal link ratio, presence of login forms, iframes, number of scripts, etc.

When the page cannot be fetched (timeout, SSL error, non‑HTML content), the HTML features are filled with a neutral fallback (`0` or `False`) and the UI indicates that HTML analysis was unavailable.

---

## Explanation Generation

`explanation.py` builds a short, human‑readable narrative based on the extracted feature values and the model’s global feature importance. It highlights:
- **Suspicious indicators** (e.g., “URL contains an IP address”, “Many subdomains”).
- **Reassuring indicators** (e.g., “HTTPS is used”, “Domain matches page title”).
- **Missing HTML analysis** when the page could not be retrieved.

---

## Security Precautions

- No JavaScript from the target site is executed.
- Requests use a short timeout (5 s) and a size limit (≈ 500 KB).
- Redirects are limited to 3 hops.
- The app never opens the URL in a browser or submits forms.
- All retrieved HTML is treated as **untrusted** data.

---

## Limitations & Future Work

- Only URL‑based and simple HTML features are used; external reputation APIs are omitted for simplicity and privacy.
- Model performance depends on the quality of the dataset; it is **not** a guarantee of safety.
- Future versions could integrate SHAP for richer per‑prediction explanations or add WHOIS/domain‑age features.

---

## Quick Commands Summary

```bash
# Environment setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Train model (creates models/ and reports/)
python train_model.py

# Run the web UI
streamlit run app.py

# Run unit tests
pytest -q tests
```

Enjoy exploring phishing detection! If you encounter any issues, check the console output for error messages or refer to the troubleshooting section in the code comments.
