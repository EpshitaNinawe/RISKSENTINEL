import pandas as pd
import os

def preprocess():

    os.makedirs("data/processed", exist_ok=True)

    cols = [
        "loan_amnt",
        "int_rate",
        "annual_inc",
        "dti",
        "revol_util",
        "installment",
        "loan_status"
    ]

    chunk_size = 10000
    chunks = []
    total_rows_needed = 50000
    rows_collected = 0

    for chunk in pd.read_csv(
        "data/raw/lendingclub.csv",
        usecols=cols,
        chunksize=chunk_size,
        low_memory=False
    ):
        chunks.append(chunk)
        rows_collected += len(chunk)

        if rows_collected >= total_rows_needed:
            break

    df = pd.concat(chunks)
    df = df.head(total_rows_needed)

    #clean percentage columns
    if df["int_rate"].dtype == "object":
        df["int_rate"] = df["int_rate"].str.replace("%", "").astype(float)

    if df["revol_util"].dtype == "object":
        df["revol_util"] = df["revol_util"].str.replace("%", "").astype(float)

    df["default"] = df["loan_status"].apply(
        lambda x: 1 if x in ["Charged Off", "Default"] else 0
    )

    df = df.drop(columns=["loan_status"])
    df = df.dropna()

    df.to_csv("data/processed/real_cleaned.csv", index=False)

    print("Cleaned dataset saved successfully.")

if __name__ == "__main__":
    preprocess()
