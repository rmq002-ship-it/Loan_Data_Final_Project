import streamlit as st
import joblib
import numpy as np
import pandas as pd
import json

# --- Bucknell Theme Config ---
BUCKNELL_ORANGE = "#E87722"
BUCKNELL_BLUE = "#003865"

st.set_page_config(page_title="Pessimistic Return Regressor", layout="wide", page_icon="📈")

# Custom CSS for Bucknell Branding and Readability
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
    /* Labels and Body text visibility */
    label, p, .stMarkdown p {{
        color: {BUCKNELL_BLUE} !important;
        font-weight: 600;
    }}
    /* Button Styling */
    div.stButton > button:first-child {{
        background-color: {BUCKNELL_ORANGE};
        color: white !important;
        border: none;
        font-weight: bold;
    }}
    div.stButton > button:first-child:hover {{
        background-color: {BUCKNELL_BLUE};
        color: white !important;
        border: 1px solid {BUCKNELL_ORANGE};
    }}
    /* Dataframe and JSON Area Focus */
    textarea {{
        border: 2px solid {BUCKNELL_BLUE} !important;
    }}
    /* Radio Buttons */
    div[data-testid="stMarkdownContainer"] p {{
        font-size: 1.1rem;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
st.title("🦬 Bucknell: Pessimistic Return Regressor")
st.write("**Bison Analytics Engine:** Input raw data below to get a standardized prediction.")
st.divider()

# --- Load Model and Scaler ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("Pessimistic_Return_model.pkl")
        # Updated to check for the scaler name you mentioned in the footer warning
        scaler = joblib.load("scaler_poly.pkl") 
        return model, scaler
    except Exception as e:
        return None, None

regressor_model, regressor_scaler = load_assets()

# --- Feature List ---
manual_features = [
   'term_months', 'fico_high', 'loan_amount interest_rate', 'loan_amount pub_rec', 
   'loan_amount home_ownership_RENT', 'loan_amount loan_purpose_home_improvement', 
   'loan_amount loan_purpose_major_purchase', 'loan_amount loan_purpose_moving', 
   'loan_amount loan_purpose_other', 'loan_amount loan_purpose_small_business', 
   'loan_amount loan_purpose_vacation', 'term_months^2', 'term_months debt_to_income', 
   'term_months employment_length_7 years', 'term_months loan_purpose_small_business', 
   'interest_rate^2', 'interest_rate debt_to_income', 'interest_rate revol_bal', 
   'interest_rate revol_util', 'interest_rate grade_B', 'interest_rate home_ownership_MORTGAGE', 
   'interest_rate verification_status_Source Verified', 'interest_rate loan_purpose_educational', 
   'annual_inc grade_E', 'annual_inc employment_length_5 years', 'annual_inc home_ownership_OWN', 
   'annual_inc verification_status_Verified', 'debt_to_income^2', 'debt_to_income pub_rec', 
   'debt_to_income grade_E', 'debt_to_income home_ownership_RENT', 'debt_to_income verification_status_Verified', 
   'debt_to_income loan_purpose_medical', 'fico_high revol_util', 'open_acc verification_status_Source Verified', 
   'open_acc loan_purpose_home_improvement', 'pub_rec employment_length_5 years', 'pub_rec employment_length_6 years', 
   'pub_rec employment_length_7 years', 'pub_rec home_ownership_OWN', 'pub_rec verification_status_Verified', 
   'pub_rec loan_purpose_credit_card', 'pub_rec loan_purpose_home_improvement', 'pub_rec loan_purpose_house', 
   'pub_rec loan_purpose_medical', 'pub_rec loan_purpose_moving', 'pub_rec loan_purpose_other', 
   'pub_rec loan_purpose_vacation', 'delinq_2yrs grade_B', 'delinq_2yrs grade_C', 'delinq_2yrs grade_E', 
   'delinq_2yrs employment_length_7 years', 'delinq_2yrs employment_length_< 1 year', 
   'delinq_2yrs loan_purpose_home_improvement', 'delinq_2yrs loan_purpose_other', 
   'delinq_2yrs loan_purpose_small_business', 'delinq_2yrs loan_purpose_vacation', 'revol_bal grade_F', 
   'revol_bal grade_G', 'revol_bal employment_length_4 years', 'revol_bal verification_status_Verified', 
   'revol_bal loan_purpose_home_improvement', 'revol_bal loan_purpose_vacation', 'revol_util grade_D', 
   'revol_util loan_purpose_house', 'revol_util loan_purpose_moving', 'revol_util loan_purpose_wedding', 
   'grade_B employment_length_10+ years', 'grade_B verification_status_Source Verified', 
   'grade_B loan_purpose_credit_card', 'grade_B loan_purpose_major_purchase', 'grade_B loan_purpose_other', 
   'grade_C employment_length_4 years', 'grade_C employment_length_6 years', 'grade_C loan_purpose_house', 
   'grade_D employment_length_3 years', 'grade_D employment_length_4 years', 'grade_D employment_length_8 years', 
   'grade_D home_ownership_MORTGAGE', 'grade_D home_ownership_RENT', 'grade_D loan_purpose_credit_card', 
   'grade_D loan_purpose_moving', 'grade_E employment_length_2 years', 'grade_E employment_length_3 years', 
   'grade_E verification_status_Source Verified', 'grade_E loan_purpose_house', 'grade_E loan_purpose_moving', 
   'grade_E loan_purpose_other', 'grade_E loan_purpose_small_business', 'grade_E loan_purpose_wedding', 
   'grade_F employment_length_10+ years', 'grade_F employment_length_5 years', 'grade_F employment_length_6 years', 
   'grade_F loan_purpose_home_improvement', 'grade_F loan_purpose_major_purchase', 'grade_F loan_purpose_medical', 
   'grade_F loan_purpose_other', 'grade_F loan_purpose_wedding', 'grade_G employment_length_2 years', 
   'grade_G employment_length_3 years', 'grade_G employment_length_5 years', 'grade_G employment_length_7 years', 
   'grade_G employment_length_8 years', 'grade_G employment_length_< 1 year', 'grade_G home_ownership_OWN', 
   'grade_G loan_purpose_debt_consolidation', 'grade_G loan_purpose_home_improvement', 'grade_G loan_purpose_house', 
   'grade_G loan_purpose_small_business', 'employment_length_10+ years home_ownership_RENT', 
   'employment_length_10+ years loan_purpose_debt_consolidation', 'employment_length_10+ years loan_purpose_medical', 
   'employment_length_2 years home_ownership_OWN', 'employment_length_2 years loan_purpose_home_improvement', 
   'employment_length_2 years loan_purpose_medical', 'employment_length_2 years loan_purpose_other', 
   'employment_length_2 years loan_purpose_renewable_energy', 'employment_length_2 years loan_purpose_small_business', 
   'employment_length_3 years home_ownership_OWN', 'employment_length_3 years home_ownership_RENT', 
   'employment_length_3 years verification_status_Verified', 'employment_length_3 years loan_purpose_house', 
   'employment_length_3 years loan_purpose_medical', 'employment_length_4 years home_ownership_MORTGAGE', 
   'employment_length_4 years home_ownership_RENT', 'employment_length_4 years loan_purpose_house', 
   'employment_length_4 years loan_purpose_moving', 'employment_length_4 years loan_purpose_other', 
   'employment_length_4 years loan_purpose_vacation', 'employment_length_5 years home_ownership_OWN', 
   'employment_length_5 years verification_status_Verified', 'employment_length_5 years loan_purpose_home_improvement', 
   'employment_length_5 years loan_purpose_moving', 'employment_length_6 years home_ownership_OWN', 
   'employment_length_6 years loan_purpose_credit_card', 'employment_length_6 years loan_purpose_vacation', 
   'employment_length_7 years home_ownership_RENT', 'employment_length_7 years loan_purpose_home_improvement', 
   'employment_length_7 years loan_purpose_other', 'employment_length_8 years home_ownership_MORTGAGE', 
   'employment_length_8 years home_ownership_OWN', 'employment_length_8 years loan_purpose_home_improvement', 
   'employment_length_8 years loan_purpose_other', 'employment_length_8 years loan_purpose_small_business', 
   'employment_length_8 years loan_purpose_vacation', 'employment_length_9 years loan_purpose_other', 
   'employment_length_< 1 year loan_purpose_major_purchase', 'employment_length_< 1 year loan_purpose_small_business', 
   'employment_length_< 1 year loan_purpose_vacation', 'home_ownership_MORTGAGE verification_status_Verified', 
   'home_ownership_OWN loan_purpose_credit_card', 'home_ownership_OWN loan_purpose_house', 
   'home_ownership_OWN loan_purpose_medical', 'home_ownership_RENT verification_status_Source Verified', 
   'home_ownership_RENT loan_purpose_home_improvement', 'home_ownership_RENT loan_purpose_major_purchase', 
   'verification_status_Source Verified loan_purpose_debt_consolidation', 
   'verification_status_Source Verified loan_purpose_moving', 'verification_status_Verified loan_purpose_credit_card', 
   'verification_status_Verified loan_purpose_house', 'verification_status_Verified loan_purpose_renewable_energy'
]

# --- UI Layout ---
# --- UI Layout ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🛠️ Input Method")
    input_mode = st.radio("Choose interaction type:", ["Single JSON Entry", "Batch CSV Upload"])

if input_mode == "Single JSON Entry":
    st.info("Paste the raw JSON below. The Bison Engine will handle scaling and prediction.")
    
    user_input_json = st.text_area("JSON Feature Vector:", 
                                   height=300, 
                                   placeholder='{"term_months": 36, "fico_high": 720, ...}')
    
    # --- This is the part that caused the error; check the spaces below! ---
    if st.button("📈 Run Single Prediction", use_container_width=True):
        if regressor_model and regressor_scaler:
            try:
                data = json.loads(user_input_json)
                raw_values = [data[f] for f in manual_features]
                input_df = pd.DataFrame([raw_values], columns=manual_features)
                
                # Apply Scaling
                scaled_values = regressor_scaler.transform(input_df)
                prediction = regressor_model.predict(scaled_values)[0]
                
                st.markdown("---")
                st.subheader("🎯 Bison Investment Analysis")
                
                # Metric and Recommendation columns
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric(label="Pessimistic Return Score", value=f"{prediction:.4f}")
                
                with res_col2:
                    if prediction > 0:
                        st.success("✅ **ACTION: APPROVE**")
                    else:
                        st.error("❌ **ACTION: DENY**")

                # Detailed text
                if prediction > 0:
                    st.info(f"The model predicts a positive return of {prediction:.4f}. This loan meets the criteria.")
                    st.balloons()
                else:
                    st.warning(f"The model predicts a negative return of {prediction:.4f}. Do not approve.")

            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.error("Model or Scaler not loaded.")

else:
    st.subheader("📂 Batch Analysis")
    uploaded_file = st.file_uploader("Upload CSV Data", type=["csv"])
    if uploaded_file and st.button("📈 Run Batch Prediction", use_container_width=True):
        if regressor_model and regressor_scaler:
            df = pd.read_csv(uploaded_file)
            missing = [f for f in manual_features if f not in df.columns]
            
            if missing:
                st.error(f"Missing {len(missing)} required features in CSV.")
            else:
                scaled_df = regressor_scaler.transform(df[manual_features])
                preds = regressor_model.predict(scaled_df)
                
                df['predicted_return'] = preds
                df['recommendation'] = np.where(df['predicted_return'] > 0, 'APPROVE', 'DENY')
                
                st.write("**Recent Predictions:**")
                st.dataframe(df[['predicted_return', 'recommendation']].head(), use_container_width=True)
                st.download_button("📥 Download Results", df.to_csv(index=False), "bison_predictions.csv")

# --- Asset Guard ---
if regressor_model is None or regressor_scaler is None:
    st.sidebar.warning("⚠️ Files Missing: 'Pessimistic_Return_model.pkl' or 'scaler_poly.pkl'")
