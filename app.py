# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.exceptions import NotFittedError

# ----- App header / branding -----
st.set_page_config(page_title="ADUSTECH Antimicrobial Predictor App", layout="centered")
st.title("🧬 ADUSTECH Antimicrobial Predictor App")
st.markdown(
    """
**Contributors:** Mujahid Nura, Abdulwahid Isah Adamu, **Sefiya Isah**  
**Purpose:** Predict resistance to Ciprofloxacin & Augmentin (multi-output) and provide rule-based antibiotic recommendations.
"""
)

st.write("---")

# ----- Sidebar info -----
with st.sidebar:
    st.header("About")
    st.write("Multi-output ML model predicts resistance for Ciprofloxacin and Augmentin using inhibition zone diameters and organism identity.")
    st.write("Model file expected: `amr_model.pkl` in the app root.")
    st.write("Ensure the model was trained with: features = [IMIPENEM, CEFTAZIDIME, GENTAMICIN, ORGANISM_ENCODED]")
    st.write("---")
    st.markdown("**How to use**:\n1. Enter zone sizes (mm)\n2. Select organism\n3. Click Predict\n4. Review prediction + recommendations")

# ----- Input fields -----
st.subheader("Enter inhibition zone diameters (mm)")
col1, col2 = st.columns(2)

with col1:
    imipenem = st.number_input("Imipenem (mm)", min_value=0.0, max_value=100.0, value=25.0, step=1.0, format="%.1f")
    ceftazidime = st.number_input("Ceftazidime (mm)", min_value=0.0, max_value=100.0, value=20.0, step=1.0, format="%.1f")

with col2:
    gentamicin = st.number_input("Gentamicin (mm)", min_value=0.0, max_value=100.0, value=20.0, step=1.0, format="%.1f")
    organism = st.selectbox(
        "Organism",
        options=["Escherichia coli", "Staphylococcus aureus", "Klebsiella pneumoniae"],
        index=0
    )

# Encode organism numerically (same as training)
organism_encoding = {"Escherichia coli": 0, "Staphylococcus aureus": 1, "Klebsiella pneumoniae": 2}
organism_encoded = organism_encoding[organism]

st.write("---")

# ----- Load model safely -----
MODEL_FILENAME = "amr_model.pkl"

def load_model(path=MODEL_FILENAME):
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}. Please upload the trained model (amr_model.pkl) to the app root.")
        return None
    try:
        m = joblib.load(path)
        return m
    except Exception as e:
        st.error("Failed to load model. This commonly happens when the model was saved with a different scikit-learn version.")
        st.caption("Retrain with scikit-learn==1.5.1 and joblib==1.4.x to fix.")
        st.exception(e)
        return None

model = load_model()

# ----- Helper: rule-based recommender -----
def recommend_alternatives(sample_dict, predicted):
    ab_zones = [
        ("Imipenem", sample_dict.get("IMIPENEM", 0.0)),
        ("Ceftazidime", sample_dict.get("CEFTAZIDIME", 0.0)),
        ("Gentamicin", sample_dict.get("GENTAMICIN", 0.0))
    ]
    sorted_by_zone = sorted(ab_zones, key=lambda x: -x[1])

    rec = {}
    cip_pred, aug_pred = int(predicted[0]), int(predicted[1])

    if cip_pred == 1:
        candidates = [name for (name, z) in sorted_by_zone]
        rec["Ciprofloxacin_alternatives"] = candidates[:2]
    if aug_pred == 1:
        candidates = [name for (name, z) in sorted_by_zone]
        rec["Augmentin_alternatives"] = candidates[:2]

    return rec

# ----- Predict button -----
if st.button("🔍 Predict"):
    if model is None:
        st.error("No model available to make predictions.")
    else:
        try:
            # include organism in the input vector
            X_user = np.array([[imipenem, ceftazidime, gentamicin, organism_encoded]])
            y_pred = model.predict(X_user)
            y_pred = y_pred[0]

            st.markdown("### Prediction Results")
            cip_text = "Resistant" if int(y_pred[0]) == 1 else "Susceptible"
            aug_text = "Resistant" if int(y_pred[1]) == 1 else "Susceptible"

            if int(y_pred[0]) == 1:
                st.error(f"Ciprofloxacin: **{cip_text}**")
            else:
                st.success(f"Ciprofloxacin: **{cip_text}**")

            if int(y_pred[1]) == 1:
                st.error(f"Augmentin: **{aug_text}**")
            else:
                st.success(f"Augmentin: **{aug_text}**")

            # recommendations
            sample = {
                "IMIPENEM": imipenem,
                "CEFTAZIDIME": ceftazidime,
                "GENTAMICIN": gentamicin
            }
            recs = recommend_alternatives(sample, y_pred)
            if recs:
                st.markdown("### 🩺 Suggested Alternatives")
                for key, val in recs.items():
                    kname = key.replace("_", " ").title()
                    st.warning(f"{kname}: {', '.join(val)}")
            else:
                st.info("No resistance predicted — all likely susceptible.")
        except Exception as e:
            st.exception(e)

st.write("---")
st.caption("Trained on Nigerian AMR dataset (E. coli, S. aureus, K. pneumoniae). Research prototype — not for direct clinical use.")
