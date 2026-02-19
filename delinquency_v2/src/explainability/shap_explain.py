import shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

def generate_shap():
    os.makedirs("models", exist_ok=True)

    model = joblib.load("models/xgb_model.pkl")
    df = pd.read_csv("data/processed/real_features.csv")

    X = df.drop(columns=["default"])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    #global summary plot
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig("models/shap_summary.png")
    plt.close()

    print("SHAP explanation saved successfully.")

if __name__ == "__main__":
    generate_shap()
