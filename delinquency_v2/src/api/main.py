from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi.responses import JSONResponse
import traceback
import shap

app = FastAPI(title="Financial Distress Engine")

# load model & SHAP explainer

xgb_model = joblib.load("models/xgb_model.pkl")
explainer = shap.TreeExplainer(xgb_model)

#input schema
class RiskInput(BaseModel):
    loan_amnt: float
    int_rate: float
    annual_inc: float
    dti: float
    revol_util: float
    installment: float

#prediction endpoint
@app.post("/predict")
def predict_risk(input_data: RiskInput):

    try:

        # Feature Engineering
        loan_income_ratio = input_data.loan_amnt / input_data.annual_inc
        emi_income_ratio = input_data.installment / input_data.annual_inc
        interest_load = input_data.int_rate * loan_income_ratio
        debt_pressure = (input_data.dti + input_data.revol_util) / 100

        financial_stress_index = (
            0.4 * emi_income_ratio +
            0.3 * loan_income_ratio +
            0.3 * debt_pressure
        )

        income_cushion = input_data.annual_inc - input_data.loan_amnt
        utilization_stress = input_data.revol_util * input_data.dti


        row = pd.DataFrame([{
            "loan_amnt": input_data.loan_amnt,
            "int_rate": input_data.int_rate,
            "installment": input_data.installment,
            "annual_inc": input_data.annual_inc,
            "dti": input_data.dti,
            "revol_util": input_data.revol_util,
            "loan_income_ratio": loan_income_ratio,
            "emi_income_ratio": emi_income_ratio,
            "interest_load": interest_load,
            "debt_pressure": debt_pressure,
            "financial_stress_index": financial_stress_index,
            "income_cushion": income_cushion,
            "utilization_stress": utilization_stress
        }])

        #model prediction
        prob = float(xgb_model.predict_proba(row)[0][1])

        # structural override Rule for high
        if (
            loan_income_ratio > 1.2 and
            emi_income_ratio > 0.05 and 
            input_data.dti > 35 and
            input_data.revol_util > 80
        ):
            prob = max(prob, 0.8)


        #risk classification
        if prob >= 0.65:
            risk_level = "HIGH"
            action = "Offer EMI restructuring or payment holiday"
        elif prob >= 0.35:
            risk_level = "MEDIUM"
            action = "Send proactive advisory message"
        else:
            risk_level = "LOW"
            action = "No action required"

        #SHAP explanation 
        shap_values = explainer.shap_values(row)

        feature_contributions = {
            col: float(shap_values[0][i])
            for i, col in enumerate(row.columns)
        }

        #top 5 most impactful features
        top_risk_drivers = dict(
            sorted(
                feature_contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]
        )

        #response
        return {
            "risk_probability": prob,
            "risk_level": risk_level,
            "recommended_action": action,
            "top_risk_drivers": top_risk_drivers
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
