from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd

# --- Create the API application ---
app_classifier = FastAPI(title="Loan Status Classifier API")

# --- Load the saved model ---
logreg_model = joblib.load("Discharge_model.pkl")

# --- Get feature names for consistent input order ---
# X_train.pkl is not available, so we need to define the features manually.
# IMPORTANT: Replace these dummy feature names with the actual feature names
# your 'Discharge_model.pkl' was trained on, in the correct order.
original_features = ['loan_amount', 'term_months', 'interest_rate', 'annual_inc', 'debt_to_income', 'fico_high', 'open_acc', 'pub_rec', 'delinq_2yrs', 'revol_bal', 'revol_util', 'grade_B', 'grade_C', 'grade_D', 'grade_E', 'grade_F', 'grade_G', 'employment_length_10+ years', 'employment_length_2 years', 'employment_length_3 years', 'employment_length_4 years', 'employment_length_5 years', 'employment_length_6 years', 'employment_length_7 years', 'employment_length_8 years', 'employment_length_9 years', 'employment_length_< 1 year', 'home_ownership_MORTGAGE', 'home_ownership_NONE', 'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT', 'verification_status_Source Verified', 'verification_status_Verified', 'loan_purpose_credit_card', 'loan_purpose_debt_consolidation', 'loan_purpose_educational', 'loan_purpose_home_improvement', 'loan_purpose_house', 'loan_purpose_major_purchase', 'loan_purpose_medical', 'loan_purpose_moving', 'loan_purpose_other', 'loan_purpose_renewable_energy', 'loan_purpose_small_business', 'loan_purpose_vacation', 'loan_purpose_wedding']

# --- Endpoint 1: Home page (just confirms the API is alive) ---
@app_classifier.get("/classifier/")
def home_classifier():
    return {"message": "Loan Status Classifier API is running"}

# --- Endpoint 2: Prediction (this is the one that does the work) ---
@app_classifier.post("/classifier/predict")
def predict_loan_status(data: dict):
    # Ensure the input data contains all expected features in the correct order
    try:
        features_array = np.array([[data[feature] for feature in original_features]])
    except KeyError as e:
        return {"error": f"Missing feature in input data: {e}"}, 400

    # Use the model to get a 'Fully Paid' probability
    # The model will directly consume the features_array without scaling
    fully_paid_prob = logreg_model.predict_proba(features_array)[0][1]

    return {
        "fully_paid_probability": round(float(fully_paid_prob), 4),
        "loan_status_prediction": "Fully Paid" if fully_paid_prob >= 0.8 else "Charged Off" # Using the determined threshold
    }
