# ===============================
# easyvisa_predict_console.py
# ===============================
import pandas as pd
import numpy as np
import joblib
import os

# ===============================
# Load model and scaler
# ===============================
try:
    model = joblib.load("best_model.pkl")
except FileNotFoundError:
    print("❌ Error: best_model.pkl not found.")
    exit()

try:
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    print("❌ Error: scaler.pkl not found.")
    exit()

# ===============================
# Expected features and defaults
# ===============================
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

# ===============================
# Utilities
# ===============================
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

# ===============================
# Console App
# ===============================
def main():
    print("🛂 EasyVisa Visa Approval Predictor (Console Version)")
    print("Type 'exit' at any prompt to quit.\n")

    while True:
        print("Choose input method:")
        print("1 - Manual Entry")
        print("2 - Upload CSV/Excel File")
        choice = input("Enter 1 or 2: ").strip()
        if choice.lower() == "exit":
            break

        input_df = None

        # -----------------------------
        # Manual Entry Mode
        # -----------------------------
        if choice == "1":
            user_input = {}
            print("\nEnter applicant details:")

            # Wage, employees, company age
            user_input["wage_yearly"] = input("Wage Offered (default=50000): ").strip()
            user_input["no_of_employees"] = input("Number of Employees (default=100): ").strip()
            user_input["company_age"] = input("Company Age (default=20): ").strip()

            # Job experience, training, full-time
            exp_val = input("Has Job Experience? (Yes/No, default=Yes): ").strip()
            train_val = input("Requires Job Training? (Yes/No, default=No): ").strip()
            full_val = input("Full-Time Position? (Yes/No, default=Yes): ").strip()

            # Encoded fields
            edu_encoded = input("Education Level Encoded (default=2): ").strip()
            cont_encoded = input("Continent Encoded (default=1): ").strip()
            region_encoded = input("Region of Employment Encoded (default=2): ").strip()

            # Convert inputs, fallback to defaults if empty
            user_input["wage_yearly"] = safe_float(user_input["wage_yearly"], FALLBACK_DEFAULTS["wage_yearly"])
            user_input["no_of_employees"] = safe_float(user_input["no_of_employees"], FALLBACK_DEFAULTS["no_of_employees"])
            user_input["company_age"] = safe_float(user_input["company_age"], FALLBACK_DEFAULTS["company_age"])

            user_input["has_job_experience"] = 1 if exp_val.lower() in ["yes", "y", "1"] else 0
            user_input["requires_job_training"] = 1 if train_val.lower() in ["yes", "y", "1"] else 0
            user_input["full_time_position"] = 1 if full_val.lower() in ["yes", "y", "1"] else 0

            user_input["education_of_employee_encoded"] = safe_float(edu_encoded, FALLBACK_DEFAULTS["education_of_employee_encoded"])
            user_input["continent_encoded"] = safe_float(cont_encoded, FALLBACK_DEFAULTS["continent_encoded"])
            user_input["region_of_employment_encoded"] = safe_float(region_encoded, FALLBACK_DEFAULTS["region_of_employment_encoded"])

            input_df = pd.DataFrame([user_input])
            input_df = ensure_features(input_df, FALLBACK_DEFAULTS)

        # -----------------------------
        # File Upload Mode
        # -----------------------------
        elif choice == "2":
            path = input("Enter CSV or Excel file path: ").strip()
            if path.lower() == "exit":
                break
            if not os.path.exists(path):
                print("❌ File not found. Try again.")
                continue

            try:
                if path.endswith(".csv"):
                    df_raw = pd.read_csv(path)
                else:
                    df_raw = pd.read_excel(path)
            except Exception as e:
                print(f"❌ Error reading file: {e}")
                continue

            input_df = encode_upload(df_raw, FALLBACK_DEFAULTS)

        else:
            print("❌ Invalid choice. Try again.")
            continue

        # -----------------------------
        # Prediction
        # -----------------------------
        y_pred, proba = predict_with_proba(model, scaler, input_df)

        input_df["Predicted_Certified"] = y_pred.astype(int)
        if proba is not None:
            input_df["Approval_Probability"] = proba

        print("\n✅ Predictions:")
        print(input_df)

        # Save to CSV
        save = input("\nDo you want to save results to CSV? (y/n): ").strip().lower()
        if save == "y":
            out_file = input("Enter output CSV file name: ").strip()
            input_df.to_csv(out_file, index=False)
            print(f"Results saved to {out_file}")

        cont = input("\nDo you want to predict again? (y/n): ").strip().lower()
        if cont != "y":
            break

if __name__ == "__main__":
    main()
