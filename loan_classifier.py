import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. Set up the page title
st.set_page_config(page_title="Loan Status Predictor")
st.title("🏦 Loan Status Classifier")
st.write("Fill in the details below to check the probability of a loan being 'Fully Paid'.")

# 2. Load the model
@st.cache_resource  # This keeps the model in memory so it doesn't reload every time
def load_model():
    return joblib.load("Discharge_model.pkl")

model = load_model()

# 3. Define the features (Your exact list)
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

# 4. Create the User Interface (Input fields)
st.sidebar.header("Input Loan Data")
input_data = {}

# We will loop through features to create input boxes
# Note: For a real app, you might want to group these or use sliders/dropdowns
for feature in original_features:
    # Defaulting to 0.0 for simplicity; you can customize these ranges
    input_data[feature] = st.sidebar.number_input(f"{feature}", value=0.0)

# 5. Prediction Logic
if st.button("📊 Predict Loan Status"):
    # Convert input dict to numpy array in the correct order
    features_array = np.array([[input_data[feature] for feature in original_features]])
    
    # Get Probability
    fully_paid_prob = model.predict_proba(features_array)[0][1]
    
    # Display Results
    st.subheader("Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Probability of 'Fully Paid'", f"{fully_paid_prob:.2%}")
    
    with col2:
        if fully_paid_prob >= 0.8:
            st.success("Prediction: **Fully Paid** ✅")
        else:
            st.error("Prediction: **Charged Off** ❌")
            
    st.info("Note: Prediction threshold is set at 0.8.")
