import pandas as pd

def build_features():
    df = pd.read_csv("data/processed/real_cleaned.csv")

    df["emi_burden_ratio"] = df["installment"] / df["annual_inc"]
    df["liquidity_risk"] = df["loan_amnt"] / df["annual_inc"]

    df["credit_pressure"] = (
        df["revol_util"] * 0.5 +
        df["dti"] * 0.5
    )

    df["financial_stress_index"] = (
        df["emi_burden_ratio"] * 0.4 +
        df["liquidity_risk"] * 0.3 +
        df["credit_pressure"] * 0.3
    )

    df.to_csv("data/processed/final_features.csv", index=False)

if __name__ == "__main__":
    build_features()
