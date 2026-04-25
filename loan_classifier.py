
Conversation with Gemini
import streamlit as st

import joblib

import numpy as np

import pandas as pd



# --- Page Configuration ---

st.set_page_config(page_title="Pessimistic Return Regressor", layout="wide")



st.title("📊 Pessimistic Return Regressor")

st.write("Upload a CSV with the required features or input data below to get a prediction.")



# --- Load Model with Caching ---

@st.cache_resource

def load_model():

    try:

        model = joblib.load("Pessimistic_Return_model.pkl")

        return model

    except Exception as e:

        st.error(f"Error loading model: {e}")

        return None



regressor_model = load_model()



# --- Feature List ---

manual_features = [

   'term_months', 'fico_high', 'loan_amount interest_rate', 'loan_amount pub_rec', 'loan_amount home_ownership_RENT', 'loan_amount loan_purpose_home_improvement', 'loan_amount loan_purpose_major_purchase', 'loan_amount loan_purpose_moving', 'loan_amount loan_purpose_other', 'loan_amount loan_purpose_small_business', 'loan_amount loan_purpose_vacation', 'term_months^2', 'term_months debt_to_income', 'term_months employment_length_7 years', 'term_months loan_purpose_small_business', 'interest_rate^2', 'interest_rate debt_to_income', 'interest_rate revol_bal', 'interest_rate revol_util', 'interest_rate grade_B', 'interest_rate home_ownership_MORTGAGE', 'interest_rate verification_status_Source Verified', 'interest_rate loan_purpose_educational', 'annual_inc grade_E', 'annual_inc employment_length_5 years', 'annual_inc home_ownership_OWN', 'annual_inc verification_status_Verified', 'debt_to_income^2', 'debt_to_income pub_rec', 'debt_to_income grade_E', 'debt_to_income home_ownership_RENT', 'debt_to_income verification_status_Verified', 'debt_to_income loan_purpose_medical', 'fico_high revol_util', 'open_acc verification_status_Source Verified', 'open_acc loan_purpose_home_improvement', 'pub_rec employment_length_5 years', 'pub_rec employment_length_6 years', 'pub_rec employment_length_7 years', 'pub_rec home_ownership_OWN', 'pub_rec verification_status_Verified', 'pub_rec loan_purpose_credit_card', 'pub_rec loan_purpose_home_improvement', 'pub_rec loan_purpose_house', 'pub_rec loan_purpose_medical', 'pub_rec loan_purpose_moving', 'pub_rec loan_purpose_other', 'pub_rec loan_purpose_vacation', 'delinq_2yrs grade_B', 'delinq_2yrs grade_C', 'delinq_2yrs grade_E', 'delinq_2yrs employment_length_7 years', 'delinq_2yrs employment_length_< 1 year', 'delinq_2yrs loan_purpose_home_improvement', 'delinq_2yrs loan_purpose_other', 'delinq_2yrs loan_purpose_small_business', 'delinq_2yrs loan_purpose_vacation', 'revol_bal grade_F', 'revol_bal grade_G', 'revol_bal employment_length_4 years', 'revol_bal verification_status_Verified', 'revol_bal loan_purpose_home_improvement', 'revol_bal loan_purpose_vacation', 'revol_util grade_D', 'revol_util loan_purpose_house', 'revol_util loan_purpose_moving', 'revol_util loan_purpose_wedding', 'grade_B employment_length_10+ years', 'grade_B verification_status_Source Verified', 'grade_B loan_purpose_credit_card', 'grade_B loan_purpose_major_purchase', 'grade_B loan_purpose_other', 'grade_C employment_length_4 years', 'grade_C employment_length_6 years', 'grade_C loan_purpose_house', 'grade_D employment_length_3 years', 'grade_D employment_length_4 years', 'grade_D employment_length_8 years', 'grade_D home_ownership_MORTGAGE', 'grade_D home_ownership_RENT', 'grade_D loan_purpose_credit_card', 'grade_D loan_purpose_moving', 'grade_E employment_length_2 years', 'grade_E employment_length_3 years', 'grade_E verification_status_Source Verified', 'grade_E loan_purpose_house', 'grade_E loan_purpose_moving', 'grade_E loan_purpose_other', 'grade_E loan_purpose_small_business', 'grade_E loan_purpose_wedding', 'grade_F employment_length_10+ years', 'grade_F employment_length_5 years', 'grade_F employment_length_6 years', 'grade_F loan_purpose_home_improvement', 'grade_F loan_purpose_major_purchase', 'grade_F loan_purpose_medical', 'grade_F loan_purpose_other', 'grade_F loan_purpose_wedding', 'grade_G employment_length_2 years', 'grade_G employment_length_3 years', 'grade_G employment_length_5 years', 'grade_G employment_length_7 years', 'grade_G employment_length_8 years', 'grade_G employment_length_< 1 year', 'grade_G home_ownership_OWN', 'grade_G loan_purpose_debt_consolidation', 'grade_G loan_purpose_home_improvement', 'grade_G loan_purpose_house', 'grade_G loan_purpose_small_business', 'employment_length_10+ years home_ownership_RENT', 'employment_length_10+ years loan_purpose_debt_consolidation', 'employment_length_10+ years loan_purpose_medical', 'employment_length_2 years home_ownership_OWN', 'employment_length_2 years loan_purpose_home_improvement', 'employment_length_2 years loan_purpose_medical', 'employment_length_2 years loan_purpose_other', 'employment_length_2 years loan_purpose_renewable_energy', 'employment_length_2 years loan_purpose_small_business', 'employment_length_3 years home_ownership_OWN', 'employment_length_3 years home_ownership_RENT', 'employment_length_3 years verification_status_Verified', 'employment_length_3 years loan_purpose_house', 'employment_length_3 years loan_purpose_medical', 'employment_length_4 years home_ownership_MORTGAGE', 'employment_length_4 years home_ownership_RENT', 'employment_length_4 years loan_purpose_house', 'employment_length_4 years loan_purpose_moving', 'employment_length_4 years loan_purpose_other', 'employment_length_4 years loan_purpose_vacation', 'employment_length_5 years home_ownership_OWN', 'employment_length_5 years verification_status_Verified', 'employment_length_5 years loan_purpose_home_improvement', 'employment_length_5 years loan_purpose_moving', 'employment_length_6 years home_ownership_OWN', 'employment_length_6 years loan_purpose_credit_card', 'employment_length_6 years loan_purpose_vacation', 'employment_length_7 years home_ownership_RENT', 'employment_length_7 years loan_purpose_home_improvement', 'employment_length_7 years loan_purpose_other', 'employment_length_8 years home_ownership_MORTGAGE', 'employment_length_8 years home_ownership_OWN', 'employment_length_8 years loan_purpose_home_improvement', 'employment_length_8 years loan_purpose_other', 'employment_length_8 years loan_purpose_small_business', 'employment_length_8 years loan_purpose_vacation', 'employment_length_9 years loan_purpose_other', 'employment_length_< 1 year loan_purpose_major_purchase', 'employment_length_< 1 year loan_purpose_small_business', 'employment_length_< 1 year loan_purpose_vacation', 'home_ownership_MORTGAGE verification_status_Verified', 'home_ownership_OWN loan_purpose_credit_card', 'home_ownership_OWN loan_purpose_house', 'home_ownership_OWN loan_purpose_medical', 'home_ownership_RENT verification_status_Source Verified', 'home_ownership_RENT loan_purpose_home_improvement', 'home_ownership_RENT loan_purpose_major_purchase', 'verification_status_Source Verified loan_purpose_debt_consolidation', 'verification_status_Source Verified loan_purpose_moving', 'verification_status_Verified loan_purpose_credit_card', 'verification_status_Verified loan_purpose_house', 'verification_status_Verified loan_purpose_renewable_energy'

]



# --- UI Layout ---

col1, col2 = st.columns([1, 2])



with col1:

    st.subheader("Input Method")

    input_mode = st.radio("Choose input method:", ["Manual Single Entry", "Batch CSV Upload"])



if input_mode == "Manual Single Entry":

    st.info("Since there are 150+ features (including interactions), please ensure your data matches the model schema.")

    

    # Example: Just showing a few inputs for brevity, but using a text area or 

    # dictionary input for the interaction terms.

    user_input_json = st.text_area("Paste Feature JSON here:", 

                                   height=300, 

                                   placeholder='{"term_months": 36, "fico_high": 720, ...}')

    

    if st.button("Predict Single"):

        if regressor_model:

            try:

                import json

                data = json.loads(user_input_json)

                features_array = np.array([[data[f] for f in manual_features]])

                prediction = regressor_model.predict(features_array)[0]

                st.success(f"Predicted Return: **{round(float(prediction), 4)}**")

            except Exception as e:

                st.error(f"Prediction Error: {e}")



else:

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file and st.button("Predict CSV"):

        if regressor_model:

            df = pd.read_csv(uploaded_file)

            # Ensure columns match

            missing = [f for f in manual_features if f not in df.columns]

            if missing:

                st.error(f"CSV missing {len(missing)} features. Example: {missing[:5]}")

            else:

                preds = regressor_model.predict(df[manual_features])

                df['predicted_return'] = preds

                st.write(df[['predicted_return']].head())

                st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv")



# --- Footer / Debug ---

if regressor_model is None:

    st.warning("⚠️ Model not loaded. Please check if 'Pessimistic_Return_model.pkl' is in the directory.") analyze each variable needed. Now find all the qunique variables so that those will be the only ones needed to by inputted in stream lit. That way it can run those polynomial columns in hte bakcgroud inherently. For example if I have age * interest rate, make it so hte user only has to put in age alone and interest rate alone on stream lit. Then the prediciton will multiply them in the bakcgorund.

To simplify your Streamlit app, we need to decouple the Raw Features (what the user provides) from the Engineered Features (what the model expects).

Since your model was trained on interaction terms (e.g., loan_amount * interest_rate) and polynomial terms (e.g., term_months^2), you only need the user to provide the "base" components.

1. Unique Base Variables
After analyzing your manual_features list, these are the 14 unique base variables the user actually needs to provide:

Category	Base Variables
Numerical	term_months, fico_high, loan_amount, interest_rate, annual_inc, debt_to_income, revol_bal, revol_util, open_acc, pub_rec, delinq_2yrs
Categorical	grade, employment_length, home_ownership, verification_status, loan_purpose
2. Implementation Strategy
We will create a function that takes these 14 inputs and programmatically builds the 150+ features required by your .pkl file.

Python
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. Define the exact feature order the model expects ---
MODEL_FEATURE_ORDER = [
    'term_months', 'fico_high', 'loan_amount interest_rate', # ... (all 150+ features here)
]

def create_feature_row(base_data):
    """
    Logic to transform 14 base inputs into the 150+ interaction features.
    base_data: dict containing the raw values
    """
    row = {}
    
    # Simple pass-throughs
    row['term_months'] = base_data['term_months']
    row['fico_high'] = base_data['fico_high']
    
    # Polynomials
    row['term_months^2'] = base_data['term_months'] ** 2
    row['interest_rate^2'] = base_data['interest_rate'] ** 2
    row['debt_to_income^2'] = base_data['debt_to_income'] ** 2
    
    # Interactions: Numerical * Numerical
    row['loan_amount interest_rate'] = base_data['loan_amount'] * base_data['interest_rate']
    row['loan_amount pub_rec'] = base_data['loan_amount'] * base_data['pub_rec']
    row['term_months debt_to_income'] = base_data['term_months'] * base_data['debt_to_income']
    # ... Add all other Num * Num interactions here ...

    # Interactions: Numerical * Categorical (One-Hot Logic)
    # Example: 'loan_amount home_ownership_RENT'
    row['loan_amount home_ownership_RENT'] = base_data['loan_amount'] if base_data['home_ownership'] == 'RENT' else 0
    row['interest_rate grade_B'] = base_data['interest_rate'] if base_data['grade'] == 'B' else 0
    
    # Interactions: Categorical * Categorical
    # Example: 'grade_B loan_purpose_credit_card'
    if base_data['grade'] == 'B' and base_data['loan_purpose'] == 'credit_card':
        row['grade_B loan_purpose_credit_card'] = 1
    else:
        row['grade_B loan_purpose_credit_card'] = 0

    # Ensure all columns exist and are in the correct order for the model
    final_df = pd.DataFrame([row])
    for col in MODEL_FEATURE_ORDER:
        if col not in final_df.columns:
            final_df[col] = 0 # Default for missing interactions
            
    return final_df[MODEL_FEATURE_ORDER]

# --- 3. Streamlit UI ---
st.title("Pessimistic Return Regressor")

with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        term = st.selectbox("Term (Months)", [36, 60])
        loan_amt = st.number_input("Loan Amount", value=10000)
        fico = st.number_input("FICO High", value=700)
        
    with col2:
        int_rate = st.number_input("Interest Rate (%)", value=12.5) / 100
        dti = st.number_input("DTI", value=15.0)
        income = st.number_input("Annual Income", value=50000)

    # ... Add inputs for Grade, Emp Length, Purpose, etc. ...
    
    submitted = st.form_submit_button("Predict")

if submitted:
    # 1. Collect inputs into a dict
    user_data = {
        "term_months": term,
        "loan_amount": loan_amt,
        "fico_high": fico,
        "interest_rate": int_rate,
        "debt_to_income": dti,
        "annual_inc": income,
        # ... include all 14 base variables ...
    }
    
    # 2. Transform
    features_for_model = create_feature_row(user_data)
    
    # 3. Predict
    prediction = regressor_model.predict(features_for_model)
    st.metric("Predicted Return", f"{prediction[0]:.4f}")
Key Considerations for your Implementation:
Naming Convention: In your manual_features list, names like loan_amount interest_rate (with a space) imply a simple multiplication.

One-Hot Encoding: For categorical interactions (like grade_B), you must check if the user's selected grade is "B". If yes, that feature is the numerical multiplier or 1 (if it's a Cat*Cat interaction).

The "Missing" Interactions: Since you have 150+ features, writing the row['...'] = ... logic manually for every line is tedious but necessary to ensure the model gets exactly what it needs.

Scaling: If you used a StandardScaler during training, you must load that .pkl as well and apply it to the final 150-feature row before model.predict().

Would you like a Python script to automatically generate the row['...'] logic by parsing your manual_features list?

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

            

    st.info("Note: Prediction threshold is set at 0.8.") turn these inputs into sliders or make the user interface more freindly and easy for people to input. It should be easy to input the variables instead of having to type them all

To make your UI more user-friendly, we need to move away from the massive list of number inputs. The best approach is to group related fields and use Dropdowns (for categorical data) and Sliders (for numerical ranges).

In the background, the code will automatically handle the "One-Hot Encoding" (turning a selection like "Mortgage" into the home_ownership_MORTGAGE = 1 the model needs).

Improved Streamlit UI Code
Python
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
        st.success(f"**High Probability of Success:** {prob:.2%}")
        st.balloons()
    elif prob >= 0.5:
        st.warning(f"**Moderate Probability:** {prob:.2%}")
    else:
        st.error(f"**Low Probability of Full Payment:** {prob:.2%}")
