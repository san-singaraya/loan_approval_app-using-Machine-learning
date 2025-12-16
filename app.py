# app.py
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="Loan Approval Prediction", layout="centered")
st.title("🏦 Loan Approval Prediction System")
st.caption("Educational purpose only")

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

df = load_data()

# --------------------------------------------------
# Dataset Overview
# --------------------------------------------------
st.markdown("## 📊 Dataset Overview")
st.write("**Shape:**", df.shape)
st.write("**Columns:**", list(df.columns))
st.dataframe(df.head())

# --------------------------------------------------
# Data Cleaning
# --------------------------------------------------
df["Dependents"] = df["Dependents"].replace("3+", 3)
df["Dependents"] = df["Dependents"].fillna(0).astype(int)

df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)
df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median(), inplace=True)
df["Credit_History"].fillna(1.0, inplace=True)

cat_cols = [
    "Gender", "Married", "Education", "Self_Employed", "Property_Area"
]

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# --------------------------------------------------
# Model Training (Random Forest)
# --------------------------------------------------
features = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Dependents",
    "Gender",
    "Married",
    "Education",
    "Self_Employed",
    "Property_Area"
]

X = df[features]
y = df["Loan_Status"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
st.success(f"✅ Model Accuracy: {accuracy:.2f}")

# --------------------------------------------------
# User Input
# --------------------------------------------------
st.markdown("## 📝 Enter Applicant Details")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_emp = st.selectbox("Self Employed", ["Yes", "No"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

dependents = st.selectbox("Dependents", [0, 1, 2, 3])

app_income = st.slider("Applicant Income", 0, 200000, 50000, step=1000)
coapp_income = st.slider("Coapplicant Income", 0, 100000, 0, step=1000)
loan_amt = st.slider("Loan Amount (in thousands)", 10, 700, 150)
loan_term = st.slider("Loan Term (Months)", 120, 480, 360)
credit_score = st.slider("Credit Score", 300, 900, 650)

# 🔑 Important change
credit_history = 1 if credit_score >= 600 else 0

gender = encoders["Gender"].transform([gender])[0]
married = encoders["Married"].transform([married])[0]
education = encoders["Education"].transform([education])[0]
self_emp = encoders["Self_Employed"].transform([self_emp])[0]
property_area = encoders["Property_Area"].transform([property_area])[0]

# --------------------------------------------------
# Prediction (PROBABILITY BASED)
# --------------------------------------------------
if st.button("🔍 Predict Loan Approval"):
    user_data = np.array([[
        app_income,
        coapp_income,
        loan_amt,
        loan_term,
        credit_history,
        dependents,
        gender,
        married,
        education,
        self_emp,
        property_area
    ]])

    user_scaled = scaler.transform(user_data)
    probability = model.predict_proba(user_scaled)[0][1]

    if probability >= 0.40:
        st.success(f"✅ Loan Approved (Probability: {probability:.2f})")
    else:
        st.error(f"❌ Loan Rejected (Probability: {probability:.2f})")

    st.info(
        "ℹ️ Loan approval strongly depends on credit history, income stability, "
        "and repayment capability. This model is trained on real bank data."
    )
