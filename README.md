# RISKSENTINEL  
### Pre-Delinquency Intervention Engine

Financial institutions often intervene only after a customer misses a payment, by which time recovery costs rise significantly and customer trust declines. Traditional collections are reactive, expensive, and operationally inefficient. The objective of this solution is to proactively identify early signs of financial distress 14–28 days before default, enabling timely and empathetic interventions. The proposed Zero-Trust Pre-Delinquency Engine leverages a hybrid AI architecture combining XGBoost for explainable risk classification and LSTM neural networks for temporal behavioral analysis. The system detects early warning signals such as salary drift, liquidity stress, EMI-to-income imbalance, and abnormal credit utilization patterns. A centralized feature store ensures consistency, retraining capability, and feature version control. The platform is built with a privacy-first framework. A Kafka-based anonymization layer tokenizes PII at ingestion, while sensitive attributes are encrypted using AES-256. MLflow and BentoML enable scalable deployment with sub-100 ms inference latency. Upon identifying high-risk cases, an orchestration engine triggers automated, empathetic interventions such as EMI restructuring or temporary payment relief. The solution aims to reduce credit losses, lower collection costs, ensure regulatory compliance (RBI/GDPR), and strengthen long-term customer relationships.

---
## Current Tech Stack

- Python
- FastAPI
- XGBoost
- SHAP (Explainability)
- Pandas / NumPy
- Scikit-learn
​
# Project Evolution

RISKSENTINEL is developed in two structured versions:

| Version | Dataset Type | Purpose |
|----------|-------------|----------|
| **v1** | Synthetic Dataset | Controlled model experimentation & pipeline validation |
| **v2** | Real Dataset (LendingClub - Kaggle) | Real-world credit risk prediction |

---
# Version 1 - delinquency_v1
The system:
- Predicts probability of default
- Classifies risk as HIGH or LOW
- Recommends intervention action
- Provides full SHAP explainability
- Maintains modular architecture for scalability

## Project Structure
```
delinquency_v1/
│
├── data/
│   ├── processed_features.csv
│   └── transactions.csv
│
├── delinq_env/              # Virtual environment
├── mlruns/                  # MLflow experiment tracking
│
├── models/
│   └── xgb_model.pkl        # Trained XGBoost model
│
├── output_ex/               # Example output screenshots (demo samples)
│   ├── high_risk_level.png
│   └── low_risk_level.png
│
├── src/
│   ├── api/
│   │   └── predict.py
│   │
│   ├── explainability/
│   │   └── shap_explainer.py
│   │
│   ├── features/
│   │   ├── build_features.py
│   │   └── advanced_features.py
│   │
│   ├── privacy/
│   │   └── tokenizer.py
│   │
│   ├── training/
│   │   ├── generate_data.py
│   │   ├── train_model.py
│   │   └── lstm_model.py
│   │
│   └── main.py
│
└── requirements.txt
```
---

## Key Features

1. Risk Prediction
   - XGBoost classifier
   - Outputs risk_score, risk_level, recommended_action

2. Explainable AI (SHAP)
   - Feature-level contribution values
   - Transparent decision breakdown
   - Audit-ready output

3. Privacy Module
   - Tokenization of PII
   - No raw user IDs exposed

4. MLflow Tracking
   - Experiment logging
   - Accuracy tracking
   - Model versioning

5. output_ex Folder
   - Contains example output screenshots
   - Demonstrates HIGH and LOW risk predictions
   - Used for demo and presentation purposes

---

## Installation

1. Create Virtual Environment
   ```
   python -m venv delinq_env
   ```
2. Activate Environment

   ```
   delinq_env\Scripts\activate
   ```

4. Install Dependencies
   ```
   pip install -r requirements.txt
    ```
---

## Running the API

From project root:
```
uvicorn src.api.predict:app --reload --port 8001
```
Open Swagger UI:
```
http://127.0.0.1:8001/docs
```
---

## Example API Input

High Risk:
```
{
  "salary": 30000,
  "emi": 28000,
  "credit_utilization": 0.97,
  "missed_payment_flag": 1
}
```

Low Risk:
```
{
  "salary": 90000,
  "emi": 8000,
  "credit_utilization": 0.20,
  "missed_payment_flag": 0
}
```
---

## Example API Output
```
{
  "risk_score": 0.84,
  "risk_level": "HIGH",
  "recommended_action": "Offer EMI restructuring",
  "explainability": {
    "method": "SHAP",
    "feature_contributions": {
      "salary": -0.12,
      "emi": 0.37,
      "credit_utilization": 0.31,
      "missed_payment_flag": 0.42,
      "emi_ratio": 0.55,
      "stress_score": 0.61
    }
  }
}
```


# Version 2 - delinquency_v2
## 📂 Project Structure

```

DELINQUENCY_V2
│
├── data/
│   ├── raw/
│   │   └── lendingclub.csv
│   └── processed/
│       ├── real_cleaned.csv
│       └── real_features.csv
│
├── models/
│   ├── xgb_model.pkl
│   ├── shap_summary.png
│   └── secret.key
│
├── output_ex/
│   ├── high_risk.jpeg
│   ├── medium_risk.jpeg
│   └── low_risk.jpeg
│
├── src/
│   ├── api/
│   │   └── main.py
│   ├── features/
│   │   └── advanced_features.py
│   ├── explainability/
│   │   └── shap_explain.py
│   ├── models/
│   │   └── train_xgb.py
│   └── security/
│
└── requirements.txt

```

---

## How It Works

### 1. Feature Engineering

The model uses raw financial inputs and engineered stress indicators:

- `loan_income_ratio`
- `emi_income_ratio`
- `interest_load`
- `debt_pressure`
- `financial_stress_index`
- `income_cushion`
- `utilization_stress`

These features capture:

- Liquidity stress  
- Debt overload  
- Credit utilization pressure  
- Income repayment burden  

---

### 2️. Model Training

Model Used: XGBoost Classifier

Why XGBoost?

- Handles nonlinear financial relationships  
- Performs well on structured tabular data  
- Robust to multicollinearity  
- Industry-proven for credit risk tasks  

---

### 3️. Explainability with SHAP

Each prediction includes:

- Top contributing features  
- SHAP impact scores  
- Transparent reasoning for the risk decision  

This makes the model:

✔ Regulator-friendly  
✔ Business explainable  
✔ Audit-ready  

---



##  Important: High-Risk Bias Issue & Solution

###  The Problem: Data Skewness

The dataset used for training had:

- Many LOW-risk borrowers  
- Fewer HIGH-risk default cases  

This imbalance caused the model to:

- Underpredict extreme high-risk profiles  
- Assign LOW probability even for clearly risky borrowers  

Example of risky borrower:

```

loan_amnt = 60000
annual_inc = 30000
dti = 45
revol_util = 95

````

Even this profile was predicted LOW due to dataset skewness.

---

## Structural Override Logic (Improvement)

To correct skewness bias, we introduced a structural stress override:

```python
if (
    loan_income_ratio > 1.2 and
    emi_income_ratio > 0.05 and
    dti > 35 and
    revol_util > 80
):
    prob = max(prob, 0.8)
````

### Why This Is Valid

This condition detects:

* Loan greater than income → severe liquidity issue
* High DTI → debt overload
* High utilization → credit pressure
* High EMI ratio → repayment stress

This ensures:

✔ Extreme structural stress is not ignored
✔ Model behaves logically in edge cases
✔ HIGH-risk borrowers are correctly classified

---

## Risk Thresholds

| Probability | Risk Level |
| ----------- | ---------- |
| ≥ 0.65      | HIGH       |
| 0.35 – 0.64 | MEDIUM     |
| < 0.35      | LOW        |

---

## Example API Request

```json
{
  "loan_amnt": 60000,
  "int_rate": 24,
  "annual_inc": 30000,
  "dti": 45,
  "revol_util": 95,
  "installment": 2000
}
```

---

## Example API Response

```json
{
  "risk_probability": 0.81,
  "risk_level": "HIGH",
  "recommended_action": "Offer EMI restructuring or payment holiday",
  "top_risk_drivers": {
      "loan_income_ratio": 1.21,
      "utilization_stress": 0.98,
      "dti": 0.76
  }
}
```

---

## Security

The project includes `secret.key` for:

* Future JWT authentication
* API request signing
* Secure deployment use

---

## ▶️ How to Run

### 1️. Activate Virtual Environment

Windows:

```
del_env\Scripts\activate
```

### 2️. Run API Server

```
python -m uvicorn src.api.main:app --port 8001 --reload
```

### 3️. Open Swagger UI

```
http://127.0.0.1:8001/docs
```

---

