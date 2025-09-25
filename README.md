#  EasyVisa ML App — Visa Approval Predictor

This project builds a **machine learning model** and a **Streamlit web app** to predict visa approval outcomes using the **EasyVisa dataset**.  
It supports both **manual entry** and **bulk file upload (CSV/Excel)** for predictions.

---

##  Features

- Train a **visa approval predictor** on the EasyVisa dataset.
- Supports:
  - Manual entry via Streamlit UI.
  - Uploading CSV/Excel files for batch predictions.
- Smart defaults for missing values.
- Encoded categorical fields (education, continent, region).
- Outputs predicted approval status + probability of approval.

---


---

## 📊 Dataset

- **Name**: `EasyVisa`
- **Target Variable**: Visa outcome (Certified / Denied)
- **Key Features**:
  - Wage Offered (Yearly)
  - Number of Employees
  - Company Age
  - Job Experience (Yes/No)
  - Requires Job Training (Yes/No)
  - Full-Time Position (Yes/No)
  - Education Level (encoded)
  - Continent (encoded)
  - Region of Employment (encoded)

---

## ⚙️ Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/your-AdeyemoOpeyemi/easyvisa-ml-app.git
   cd easyvisa-ml-app


