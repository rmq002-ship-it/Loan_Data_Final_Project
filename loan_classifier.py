import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --- Page Config ---
st.set_page_config(page_title="Loan Status Predictor", layout="centered")
st.title("🏦 Loan Status Classifier")
st.write("Easily predict the likelihood of a loan being fully paid.")

# --- Load Model ---
@st.cache_resource
def load_model():
    return joblib.load("Discharge_model.pkl")

model = load_model()

# --- Helper Logic: Mapping Categories to the Model's Features ---
# This list is exactly what your model expects
original_features = [
    'loan_amount', 'term_months', 'interest_rate', 'annual_inc', 'debt_to_income', 
    'fico_high', 'open_acc', 'pub_rec', 'delinq_2yrs', 'revol_bal', 'revol_util', 
    'grade_B', 'grade_C', 'grade_D', 'grade_E', 'grade_F', 'grade_G', 
    'employment_length_10+ years', 'employment_length_2 years', 'employment_length_3 years', 
    'employment_length_4 years', 'employment_length_5 years', 'employment_length_6 years', 
    'employment_length_7 years', 'employment_length_8 years', 'employment_length_9 years', 
    'employment_length_< 1 year', 'home_ownership_MORTGAGE', 'home_ownership_NONE', 
    'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT', 
    'verification_status_Source Verified', 'verification_status_Verified', 
    'loan_purpose_credit_card', 'loan_purpose_debt_consolidation', 'loan_purpose_educational', 
    'loan_purpose_home_improvement', 'loan_purpose_house', 'loan_purpose_major_purchase', 
    'loan_purpose_medical', 'loan_purpose_moving', 'loan_purpose_other', 
    'loan_purpose_renewable_energy', 'loan_purpose_small_business', 'loan_purpose_vacation', 
    'loan_purpose_wedding'
]

# --- 1. User Interface: Organized Inputs ---
st.subheader("📝 Applicant & Loan Details")

col1, col2 = st.columns(2)

with col1:
    loan_amount = st.slider("Loan Amount ($)", 500, 40000, 10000, step=500)
    term = st.selectbox("Term", options=[36, 60], format_func=lambda x: f"{x} Months")
    int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 12.0, step=0.1)
    annual_inc = st.number_input("Annual Income ($)", value=50000, step=1000)
    dti = st.slider("Debt-to-Income Ratio (DTI)", 0.0, 40.0, 15.0)

with col2:
    fico = st.slider("FICO Score (High)", 300, 850, 700)
    rev_util = st.slider("Revolving Utilization (%)", 0.0, 100.0, 30.0)
    open_acc = st.number_input("Open Credit Lines", 0, 50, 10)
    pub_rec = st.number_input("Public Records", 0, 10, 0)
    delinq = st.number_input("Delinquencies (Last 2 yrs)", 0, 10, 0)
    revol_bal = st.number_input("Total Revolving Balance ($)", value=5000)

st.divider()

col3, col4 = st.columns(2)

with col3:
    grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
    emp_len = st.selectbox("Employment Length", [
        "< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", 
        "6 years", "7 years", "8 years", "9 years", "10+ years"
    ])

with col4:
    home = st.selectbox("Home Ownership", ["MORTGAGE", "RENT", "OWN", "OTHER", "NONE"])
    purpose = st.selectbox("Loan Purpose", [
        "credit_card", "debt_consolidation", "educational", "home_improvement", 
        "house", "major_purchase", "medical", "moving", "other", 
        "renewable_energy", "small_business", "vacation", "wedding"
    ])
    ver_stat = st.selectbox("Verification Status", ["Not Verified", "Source Verified", "Verified"])

# --- 2. Data Transformation (Automatic Encoding) ---
# Initialize all features to 0
encoded_data = {feat: 0.0 for feat in original_features}

# Fill Numerical values
encoded_data['loan_amount'] = float(loan_amount)
encoded_data['term_months'] = float(term)
encoded_data['interest_rate'] = float(int_rate)
encoded_data['annual_inc'] = float(annual_inc)
encoded_data['debt_to_income'] = float(dti)
encoded_data['fico_high'] = float(fico)
encoded_data['open_acc'] = float(open_acc)
encoded_data['pub_rec'] = float(pub_rec)
encoded_data['delinq_2yrs'] = float(delinq)
encoded_data['revol_bal'] = float(revol_bal)
encoded_data['revol_util'] = float(rev_util)

# Set Categorical flags (One-Hot Encoding logic)
if f"grade_{grade}" in encoded_data: encoded_data[f"grade_{grade}"] = 1.0
if f"employment_length_{emp_len}" in encoded_data: encoded_data[f"employment_length_{emp_len}"] = 1.0
if f"home_ownership_{home}" in encoded_data: encoded_data[f"home_ownership_{home}"] = 1.0
if f"verification_status_{ver_stat}" in encoded_data: encoded_data[f"verification_status_{ver_stat}"] = 1.0
if f"loan_purpose_{purpose}" in encoded_data: encoded_data[f"loan_purpose_{purpose}"] = 1.0

# --- 3. Prediction ---
if st.button("📊 Predict Probability", use_container_width=True):
    # Prepare array in exact order
    input_vector = np.array([[encoded_data[f] for f in original_features]])
    
    # Get Probability
    prob = model.predict_proba(input_vector)[0][1]
    
    st.markdown("---")
    st.subheader("Prediction Result")
    
    # Visual feedback based on probability
    if prob >= 0.8:
        st.success(f"**High Probability of Fully Paying the Loan**")
        st.balloons()
    else:
        st.error(f"**Will Charge Off the Loan**")
