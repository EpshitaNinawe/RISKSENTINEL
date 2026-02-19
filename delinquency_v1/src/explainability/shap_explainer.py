import shap
import joblib
import numpy as np
import pandas as pd

#load trained model
model = joblib.load("models/xgb_model.pkl")

FEATURE_NAMES = [
    "salary",
    "emi",
    "credit_utilization",
    "missed_payment_flag",
    "emi_ratio",
    "stress_score"
]

#create SHAP explainer
explainer = shap.TreeExplainer(model)

def explain_prediction(feature_array):
    """
    feature_array: numpy array of shape (1, n_features)
    """
    shap_values = explainer.shap_values(feature_array)

    explanation = dict(zip(
        FEATURE_NAMES,
        shap_values[0].tolist()
    ))

    return explanation
