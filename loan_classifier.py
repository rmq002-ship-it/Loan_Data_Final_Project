import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --- Updated Bucknell Theme Config ---
BUCKNELL_ORANGE = "#E87722"
BUCKNELL_BLUE = "#003865"

st.markdown(f"""
    <style>
    /* Main background */
    .stApp {{
        background-color: #FFFFFF;
    }}
    
    /* Force all widget labels to be visible (Dark Blue or Black) */
    label, .stMarkdown p, .stSelectbox label, .stSlider label {{
        color: {BUCKNELL_BLUE} !important;
        font-weight: 600 !important;
    }}

    /* Fix the text inside the sliders/inputs that was invisible */
    div[data-baseweb="slider"] div {{
        color: {BUCKNELL_ORANGE} !important;
    }}
    
    /* Headers */
    h1, h2, h3 {{
        color: {BUCKNELL_BLUE} !important;
        font-family: 'Georgia', serif;
    }}

    /* Button Styling - Orange with Blue hover */
    div.stButton > button:first-child {{
        background-color: {BUCKNELL_ORANGE};
        color: white !important;
        border: none;
        padding: 0.5rem 2rem;
    }}
    
    div.stButton > button:first-child:hover {{
        background-color: {BUCKNELL_BLUE};
        color: white !important;
    }}

    /* Customize the slider track color to Bucknell Blue */
    .stSlider [data-baseweb="slider"] > div > div {{
        background: {BUCKNELL_BLUE} !important;
    }}
    </style>
    """, unsafe_allow_html=True)
st.markdown(f"""
    <style>
    .stApp {{
        background-color: #FFFFFF;
    }}
    
    /* Title and Subheaders */
    h1, h2, h3 {{
        color: {BUCKNELL_BLUE} !important;
        font-family: 'Georgia', serif;
    }}
    
    /* Force ALL labels and main text to Bucknell Blue */
    label, p, .stMarkdown p {{
        color: {BUCKNELL_BLUE} !important;
        font-weight: 600 !important;
    }}

    /* FIX: Force text inside success/warning/info boxes to be dark blue */
    div[data-testid="stNotification"] p {{
        color: {BUCKNELL_BLUE} !important;
        font-weight: 700 !important;
    }}

    /* Button Styling */
    div.stButton > button:first-child {{
        background-color: {BUCKNELL_ORANGE};
        color: white !important;
        border: none;
        font-weight: bold;
        padding: 0.5rem 2rem;
    }}
    
    div.stButton > button:first-child:hover {{
        background-color: {BUCKNELL_BLUE};
        color: white !important;
    }}

    /* Textarea border visibility */
    textarea {{
        border: 2px solid {BUCKNELL_BLUE} !important;
        color: #333333 !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- Header Section ---
st.title("🦬 Bucknell Loan Status Classifier")
st.write(f"**Ray Bucknell!** Use this tool to predict the likelihood of a loan being fully paid.")
st.divider()

# --- Load Model and Scaler ---
@st.cache_resource
def load_assets():
    # Ensure these filenames match your actual saved files!
    model = joblib.load("Discharge_model.pkl")
    scaler = joblib.load("scaler.pkl") 
    return model, scaler

# Wrapping in try-except in case files aren't found during initial setup
try:
    model, scaler = load_assets()
except FileNotFoundError:
    st.error("⚠️ Model or Scaler files not found. Please ensure 'Discharge_model.pkl' and 'scaler.pkl' are in the directory.")
    st.stop()

# --- Feature List ---
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

# --- 1. User Interface ---
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

st.markdown("---")
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

# --- 2. Data Transformation ---
encoded_data = {feat: 0.0 for feat in original_features}
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

# Set Categorical flags
if f"grade_{grade}" in encoded_data: encoded_data[f"grade_{grade}"] = 1.0
if f"employment_length_{emp_len}" in encoded_data: encoded_data[f"employment_length_{emp_len}"] = 1.0
if f"home_ownership_{home}" in encoded_data: encoded_data[f"home_ownership_{home}"] = 1.0
if f"verification_status_{ver_stat}" in encoded_data: encoded_data[f"verification_status_{ver_stat}"] = 1.0
if f"loan_purpose_{purpose}" in encoded_data: encoded_data[f"loan_purpose_{purpose}"] = 1.0

# --- 3. Prediction ---
if st.button("📊 Calculate Bison Score", use_container_width=True):
    input_vector = np.array([[encoded_data[f] for f in original_features]])
    input_df = pd.DataFrame(input_vector, columns=original_features)
    scaled_vector = scaler.transform(input_df)
    
    prob = model.predict_proba(scaled_vector)[0][1]
    
    st.markdown("---")
    st.subheader("Results")
    
    if prob >= 0.8:
        st.success(f"**Approved! The model predicts the loan will be FULLY PAID (Prob: {prob:.2%})**")
        st.balloons()
    else:
        st.warning(f"**High Risk: The model predicts the loan may CHARGE OFF (Prob: {prob:.2%})**")
