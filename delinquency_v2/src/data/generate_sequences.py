import numpy as np
import pandas as pd
import os

def generate_sequences(num_users=1000):
    os.makedirs("data/processed", exist_ok=True)

    sequences = []

    for _ in range(num_users):
        base = np.random.normal(500, 100)
        trend = np.random.uniform(-5, 10)

        seq = [base + i*trend + np.random.normal(0, 50)
               for i in range(30)]

        sequences.append(seq)

    sequences = np.array(sequences)

    np.save("data/processed/transaction_sequences.npy", sequences)
    print("Transaction sequences generated.")

if __name__ == "__main__":
    generate_sequences()
