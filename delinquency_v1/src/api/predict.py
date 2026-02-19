import joblib
import numpy as np
from fastapi import FastAPI
from src.explainability.shap_explainer import explain_prediction

app = FastAPI()

model = joblib.load("models/xgb_model.pkl")

@app.post("/predict")
def predict(data: dict):
    salary = data["salary"]
    emi = data["emi"]
    credit_util = data["credit_utilization"]
    missed = data["missed_payment_flag"]

    emi_ratio = emi / salary
    stress_score = emi_ratio*0.5 + credit_util*0.3 + missed*0.2

    features = np.array([[salary, emi, credit_util, missed, emi_ratio, stress_score]])

    risk_prob = model.predict_proba(features)[0][1]

    risk_level = "HIGH" if risk_prob > 0.75 else "LOW"

    #intervention trigger
    if risk_level == "HIGH":
        intervention = "Offer EMI restructuring"
    else:
        intervention = "No action required"

    shap_explanation = explain_prediction(features)

    return {
        "risk_score": float(risk_prob),
        "risk_level": risk_level,
        "recommended_action": intervention,
        "explainability": {
            "method": "SHAP",
            "feature_contributions": shap_explanation
        }
    }
