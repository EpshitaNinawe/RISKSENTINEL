import pandas as pd
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train():
    df = pd.read_csv("data/processed_features.csv")

    X = df[[
        "salary",
        "emi",
        "credit_utilization",
        "missed_payment_flag",
        "emi_ratio",
        "stress_score"
    ]]

    y = df["will_default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBClassifier()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.start_run()
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "xgb_model")
    mlflow.end_run()

    joblib.dump(model, "models/xgb_model.pkl")

    print("Model trained. Accuracy:", acc)

if __name__ == "__main__":
    train()
