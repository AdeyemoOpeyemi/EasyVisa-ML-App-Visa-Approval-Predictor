
# app.py — EasyVisa Visa Approval Predictor
# Supports: manual entry, CSV/Excel upload, smart defaults, country encoding
# Requirements: streamlit, pandas, numpy, joblib, scikit-learn

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="EasyVisa Predictor", page_icon="🛂", layout="centered")

# -----------------------------
# Core configuration
# -----------------------------
EXPECTED_FEATURES = [
    "wage_yearly", "no_of_employees", "company_age",
    "has_job_experience", "requires_job_training", "full_time_position",
    "education_of_employee_encoded", "continent_encoded", "region_of_employment_encoded"
]

FALLBACK_DEFAULTS = {
    "wage_yearly": 50000.0,
    "no_of_employees": 100,
    "company_age": 20,
    "has_job_experience": 1,
    "requires_job_training": 0,
    "full_time_position": 1,
    "education_of_employee_encoded": 2,
    "continent_encoded": 1,
    "region_of_employment_encoded": 2
}

# -----------------------------
# Utilities
# -----------------------------
@st.cache_resource
def load_model_and_meta():
    try:
        model = joblib.load("best_model.pkl")
    except Exception:
        st.error("❌ 'best_model.pkl' not found. Please place it in the same folder as app.py.")
        st.stop()

    try:
        scaler = joblib.load("scaler.pkl")
    except Exception:
        st.error("❌ 'scaler.pkl' not found. Please place it in the same folder as app.py.")
        st.stop()

    return model, scaler

def safe_float(value, default):
    try:
        return float(value) if pd.notna(value) else default
    except:
        return default

def ensure_features(df, defaults):
    for col in EXPECTED_FEATURES:
        if col not in df.columns:
            df[col] = defaults[col]
    return df[EXPECTED_FEATURES]

def build_template_csv():
    return pd.DataFrame({
        "wage_yearly": [60000, 85000],
        "no_of_employees": [150, 5000],
        "company_age": [10, 25],
        "has_job_experience": [1, 0],
        "requires_job_training": [0, 1],
        "full_time_position": [1, 1],
        "education_of_employee_encoded": [2, 3],
        "continent_encoded": [1, 2],
        "region_of_employment_encoded": [2, 1]
    })

def encode_upload(df_raw, defaults):
    df = df_raw.copy()
    for col in EXPECTED_FEATURES:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: safe_float(x, defaults[col]))
        else:
            df[col] = defaults[col]
    return ensure_features(df, defaults)

def predict_with_proba(model, scaler, X):
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    proba = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else None
    return y_pred, proba

# -----------------------------
# Load model and scaler
# -----------------------------
model, scaler = load_model_and_meta()
DEFAULTS = FALLBACK_DEFAULTS.copy()

st.title("🛂 EasyVisa Visa Approval Predictor")
st.write("Predict visa approval using manual inputs or file upload. Missing fields are auto-filled.")

# -----------------------------
# Input mode
# -----------------------------
mode = st.radio("Choose input method:", ["Manual entry", "Upload file"], horizontal=True)

# -----------------------------
# Manual entry
# -----------------------------
if mode == "Manual entry":
    st.subheader("Enter applicant details")
    st.caption("Uncheck a field to use default values.")

    col1, col2 = st.columns(2)

    with col1:
        use_wage = st.checkbox("Wage Offered (Yearly)", value=True)
        wage_val = st.number_input("Wage (USD)", value=DEFAULTS["wage_yearly"]) if use_wage else DEFAULTS["wage_yearly"]

        use_exp = st.checkbox("Has Job Experience", value=True)
        exp_val = st.selectbox("Job Experience", ["Yes", "No"]) if use_exp else ("Yes" if DEFAULTS["has_job_experience"] == 1 else "No")
        exp_encoded = 1 if exp_val == "Yes" else 0

        use_train = st.checkbox("Requires Job Training", value=True)
        train_val = st.selectbox("Training Required", ["No", "Yes"]) if use_train else ("No" if DEFAULTS["requires_job_training"] == 0 else "Yes")
        train_encoded = 1 if train_val == "Yes" else 0

    with col2:
        use_emp = st.checkbox("Number of Employees", value=True)
        emp_val = st.number_input("Employees", value=DEFAULTS["no_of_employees"]) if use_emp else DEFAULTS["no_of_employees"]

        use_age = st.checkbox("Company Age", value=True)
        age_val = st.slider("Company Age (years)", 0, 300, int(DEFAULTS["company_age"])) if use_age else DEFAULTS["company_age"]

        use_full = st.checkbox("Full-Time Position", value=True)
        full_val = st.selectbox("Full-Time", ["Yes", "No"]) if use_full else ("Yes" if DEFAULTS["full_time_position"] == 1 else "No")
        full_encoded = 1 if full_val == "Yes" else 0

    st.markdown("### Encoded Inputs")
    edu_encoded = st.slider("Education Level (Encoded)", 0, 5, DEFAULTS["education_of_employee_encoded"])
    cont_encoded = st.slider("Continent (Encoded)", 0, 5, DEFAULTS["continent_encoded"])
    region_encoded = st.slider("Region of Employment (Encoded)", 0, 5, DEFAULTS["region_of_employment_encoded"])

    row = pd.DataFrame([{
        "wage_yearly": float(wage_val),
        "no_of_employees": float(emp_val),
        "company_age": float(age_val),
        "has_job_experience": int(exp_encoded),
        "requires_job_training": int(train_encoded),
        "full_time_position": int(full_encoded),
        "education_of_employee_encoded": int(edu_encoded),
        "continent_encoded": int(cont_encoded),
        "region_of_employment_encoded": int(region_encoded)
    }])

    row = ensure_features(row, DEFAULTS)

    if st.button("Predict visa outcome"):
        y_pred, proba = predict_with_proba(model, scaler, row)
        label = int(y_pred[0])
        approved_prob = float(proba[0]) if proba is not None else None

        st.markdown("---")
        st.success("Prediction: Certified ✅" if label == 1 else "Prediction: Denied ❌")
        if approved_prob is not None:
            st.info(f"Estimated approval probability: {approved_prob:.2%}")
        st.caption("Probability is an estimate based on the model and available features.")

# -----------------------------
# File upload
# -----------------------------
else:
    st.subheader("Upload CSV or Excel")
    st.write("Include any of the expected columns. Missing fields are auto-filled.")

    template = build_template_csv()
    csv_data = template.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download input template", data=csv_data, file_name="easyvisa_input_template.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload file", type=["csv", "xlsx"])
    if uploaded:
        df_raw = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)

        st.write("Preview of uploaded data:")
        st.dataframe(df_raw.head())

        X = encode_upload(df_raw, DEFAULTS)
        y_pred, proba = predict_with_proba(model, scaler, X)

        out = df_raw.copy()
        out["Predicted_Certified"] = y_pred.astype(int)
        if proba is not None:
            out["Approval_Probability"] = proba

        st.success("✅ Predictions complete")
        st.dataframe(out)

        csv_out = out.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download predictions", data=csv_out, file_name="easyvisa_predictions.csv", mime="text/csv")

# -----------------------------
# Help section
# -----------------------------
with st.expander("ℹ️ How this app works"):
    st.markdown("""
- You can enter data manually or upload a file for bulk predictions.
- Missing fields are filled with smart defaults based on training data.
- Encoded values""")