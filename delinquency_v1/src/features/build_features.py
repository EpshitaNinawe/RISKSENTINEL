import pandas as pd
from src.privacy.tokenizer import tokenize_pii

def build_features():
    df = pd.read_csv("data/transactions.csv")

    # Tokenize user_id
    df["user_token"] = df["user_id"].apply(tokenize_pii)

    # Feature Engineering
    df["emi_ratio"] = df["emi"] / df["salary"]
    df["stress_score"] = (
        df["emi_ratio"] * 0.5 +
        df["credit_utilization"] * 0.3 +
        df["missed_payment_flag"] * 0.2
    )

    df.to_csv("data/processed_features.csv", index=False)

if __name__ == "__main__":
    build_features()
