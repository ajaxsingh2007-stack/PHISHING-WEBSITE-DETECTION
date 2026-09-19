# app.py
"""
Streamlit frontend for the Explainable Phishing Website Detector.
"""
import streamlit as st
import joblib
import numpy as np

from feature_extractor import extract_features
from explanation import generate_explanation

# ----------------------------------------------------------------------
st.set_page_config(page_title="PhishDetect", layout="centered")

@st.cache_resource
def load_model():
    clf = joblib.load("models/phishing_model.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return clf, feature_names

clf, feature_names = load_model()

st.title("🔐 Explainable Phishing Website Detector")
st.caption("Analyze a website URL using a Random Forest classifier and see why the model made its decision.")

# ----------------------------------------------------------------------
# User input
# ----------------------------------------------------------------------
url_input = st.text_input(
    "Enter website URL",
    placeholder="https://example.com",
    help="Paste any URL you want to check. The system will *not* open the site in a browser.",
)

if st.button("Analyze URL"):
    if not url_input.strip():
        st.error("Please enter a URL.")
    else:
        try:
            # Extract features (matches training order)
            feats = extract_features(url_input)
            X = np.array([feats[name] for name in feature_names]).reshape(1, -1)
            prob_phish = clf.predict_proba(X)[0, 1]
            pred_label = "Potential Phishing" if prob_phish >= 0.5 else "Likely Legitimate"
            if feats.get("html_status") == "unavailable":
                # We still have a probability, but we note the lack of HTML analysis
                st.warning("HTML analysis was unavailable; prediction is based on URL‑only features.")
            st.subheader("Result")
            st.success(f"**{pred_label}** – probability = {prob_phish:.2%}")
            # Explanation
            st.subheader("Explanation")
            st.write(generate_explanation(feats, prob_phish))
            # Feature table
            st.subheader("Extracted Features (used by the model)")
            import pandas as pd
            df_feat = pd.DataFrame({"Feature": feature_names, "Value": [feats[n] for n in feature_names]})
            st.dataframe(df_feat, hide_index=True)
        except Exception as e:
            st.error(f"An error occurred while processing the URL: {e}")

# Disclaimer
st.markdown(
    """
    **Disclaimer:**\
    This tool provides an *estimated* classification based on statistical patterns.\
    It does **not** guarantee that a website is safe or malicious.\
    Always exercise caution, especially before entering credentials or payment information.
    """,
    unsafe_allow_html=True,
)
