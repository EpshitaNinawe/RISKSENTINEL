import pandas as pd

def build_advanced_features():

    df = pd.read_csv("data/processed/real_cleaned.csv")

    df["loan_income_ratio"] = df["loan_amnt"] / df["annual_inc"]
    df["emi_income_ratio"] = df["installment"] / df["annual_inc"]
    df["interest_load"] = df["int_rate"] * df["loan_income_ratio"]
    df["debt_pressure"] = (df["dti"] + df["revol_util"]) / 100

    df["financial_stress_index"] = (
        0.4 * df["emi_income_ratio"] +
        0.3 * df["loan_income_ratio"] +
        0.3 * df["debt_pressure"]
    )

    df["income_cushion"] = df["annual_inc"] - df["loan_amnt"]
    df["utilization_stress"] = df["revol_util"] * df["dti"]

    df.to_csv("data/processed/real_features.csv", index=False)

    print("Advanced features created successfully.")

if __name__ == "__main__":
    build_advanced_features()
