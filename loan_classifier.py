from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd

# --- Create the API application ---
app_classifier = FastAPI(title="Loan Status Classifier API")

# --- Load the saved model ---
logreg_model = joblib.load("Discharge_model.pkl")

# --- Get feature names for consistent input order ---
# Assuming X_train.pkl contains the feature names in the correct order
original_features = joblib.load('X_train.pkl').columns

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
