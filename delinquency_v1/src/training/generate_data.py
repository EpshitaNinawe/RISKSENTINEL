import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()

def generate_dataset(n=2000):
    data = []

    for _ in range(n):
        salary = random.randint(20000, 80000)
        emi = random.randint(2000, 25000)
        credit_util = random.uniform(0.2, 1.0)
        missed_payment = random.choice([0, 1])

        #default logic simulation
        risk = 1 if (emi > salary * 0.5 or credit_util > 0.85 or missed_payment == 1) else 0

        data.append([
            fake.uuid4(),
            salary,
            emi,
            credit_util,
            missed_payment,
            risk
        ])

    df = pd.DataFrame(data, columns=[
        "user_id",
        "salary",
        "emi",
        "credit_utilization",
        "missed_payment_flag",
        "will_default"
    ])

    df.to_csv("data/transactions.csv", index=False)

if __name__ == "__main__":
    generate_dataset()
