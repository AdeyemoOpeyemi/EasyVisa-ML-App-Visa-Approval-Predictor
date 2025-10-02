import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="EasyVisa Predictor", page_icon="🛂", layout="centered")

# -----------------------------
# Core configuration
# -----------------------------
ALL_FEATURES = [
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
def safe_float(value, default):
    try:
        return float(value) if pd.notna(value) else default
    except:
        return default

def ensure_features(df, defaults, selected_features):
    for col in selected_features:
        if col not in df.columns:
            df[col] = defaults.get(col, 0)
    return df[selected_features]

def encode_upload(df_raw, defaults, selected_features):
    df = df_raw.copy()
    for col in selected_features:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: safe_float(x, defaults.get(col, 0)))
        else:
            df[col] = defaults.get(col, 0)
    return ensure_features(df, defaults, selected_features)

def predict_with_proba(model, scaler, X):
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    proba = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else None
    return y_pred, proba

# -----------------------------
# Load model and scaler
# -----------------------------
@st.cache_resource
def load_model_and_meta():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))

    model_path = os.path.join(models_dir, "best_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")

    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at {model_path}. Please place 'best_model.pkl' in the 'models/' folder.")
        st.stop()

    if not os.path.exists(scaler_path):
        st.error(f"❌ Scaler not found at {scaler_path}. Please place 'scaler.pkl' in the 'models/' folder.")
        st.stop()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_model_and_meta()
DEFAULTS = FALLBACK_DEFAULTS.copy()

# -----------------------------
# Feature selection
# -----------------------------
st.sidebar.header("Select Features to Use")
selected_features = st.sidebar.multiselect(
    "Pick the features to include",
    options=ALL_FEATURES,
    default=ALL_FEATURES
)

if not selected_features:
    st.warning("Please select at least one feature to continue.")
    st.stop()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🛂 EasyVisa Visa Approval Predictor")
st.write("Predict visa approval using manual inputs or file upload. Missing fields are auto-filled.")

mode = st.radio("Choose input method:", ["Manual entry", "Upload file"], horizontal=True)

# -----------------------------
# Manual entry
# -----------------------------
if mode == "Manual entry":
    st.subheader("Enter applicant details")
    row_data = {}
    col1, col2 = st.columns(2)

    for col in selected_features:
        if col == "wage_yearly":
            row_data[col] = col1.number_input("Wage Offered (Yearly USD)", value=DEFAULTS[col])
        elif col == "no_of_employees":
            row_data[col] = col1.number_input("Number of Employees", value=DEFAULTS[col])
        elif col == "company_age":
            row_data[col] = col1.slider("Company Age (Years)", 0, 300, int(DEFAULTS[col]))
        elif col == "has_job_experience":
            val = col2.selectbox("Has Job Experience", ["Yes", "No"], index=DEFAULTS[col])
            row_data[col] = 1 if val == "Yes" else 0
        elif col == "requires_job_training":
            val = col2.selectbox("Requires Job Training", ["Yes", "No"], index=DEFAULTS[col])
            row_data[col] = 1 if val == "Yes" else 0
        elif col == "full_time_position":
            val = col2.selectbox("Full-Time Position", ["Yes", "No"], index=DEFAULTS[col])
            row_data[col] = 1 if val == "Yes" else 0
        elif col in ["education_of_employee_encoded", "continent_encoded", "region_of_employment_encoded"]:
            row_data[col] = col2.slider(col.replace("_", " ").title(), 0, 5, DEFAULTS[col])

    row = pd.DataFrame([row_data])
    row = ensure_features(row, DEFAULTS, selected_features)

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
    uploaded = st.file_uploader("Upload file", type=["csv", "xlsx"])
    if uploaded:
        df_raw = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
        st.write("Preview of uploaded data:")
        st.dataframe(df_raw.head())

        # Align selected features with uploaded file
        df_for_model = encode_upload(df_raw, DEFAULTS, selected_features)
        y_pred, proba = predict_with_proba(model, scaler, df_for_model)

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
- You can dynamically select which features to include both in manual entry and file upload.
- Ensure 'best_model.pkl' and 'scaler.pkl' are in the 'models/' folder.
""")
