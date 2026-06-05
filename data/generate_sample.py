"""
data/generate_sample.py
------------------------
Generates a synthetic IEEE-CIS-like fraud dataset for testing
without needing the Kaggle download.

Usage:
    python generate_sample.py --rows 10000 --output sample_transactions.csv

This produces a CSV with the same column structure as train_transaction.csv
so all pipeline scripts work without modification.
"""

import argparse
import numpy as np
import pandas as pd
import random

parser = argparse.ArgumentParser(description="Synthetic fraud data generator")
parser.add_argument("--rows",   type=int, default=10000,
                    help="Number of rows to generate (default: 10000)")
parser.add_argument("--output", default="sample_transactions.csv",
                    help="Output CSV filename")
parser.add_argument("--fraud_rate", type=float, default=0.035,
                    help="Fraction of fraudulent transactions (default: 0.035)")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)

N = args.rows
fraud_mask = np.random.random(N) < args.fraud_rate

# TransactionDT — seconds offset from a reference point, sorted
tx_dt = np.sort(np.random.randint(86400, 86400 * 180, N))

# TransactionAmt — fraud transactions tend to be higher value
amt_legit = np.random.lognormal(mean=4.0, sigma=1.2, size=N)
amt_fraud  = np.random.lognormal(mean=5.5, sigma=1.5, size=N)
tx_amt = np.where(fraud_mask, amt_fraud, amt_legit).round(2)

# Card fields
card1 = np.random.randint(1000, 18000, N)
card2 = np.random.choice([111., 222., 321., 150., 555., np.nan], N)
card3 = np.random.choice([150., 185., 144., np.nan], N)
card4 = np.random.choice(["visa", "mastercard", "american express",
                           "discover", None], N,
                          p=[0.5, 0.3, 0.1, 0.08, 0.02])
card5 = np.random.choice([102., 117., 226., 224., np.nan], N)
card6 = np.random.choice(["debit", "credit", "debit or credit",
                           "charge card", None], N,
                          p=[0.5, 0.35, 0.08, 0.05, 0.02])

# Address
addr1 = np.random.choice(list(range(100, 500)) + [np.nan], N)
addr2 = np.random.choice(list(range(10, 102)) + [np.nan], N)
dist1 = np.where(np.random.random(N) < 0.3, np.nan,
                 np.random.exponential(scale=50, size=N).round(1))

# Product code
product_cd = np.random.choice(["W", "H", "C", "S", "R"], N,
                                p=[0.6, 0.15, 0.1, 0.1, 0.05])

# Email domains
legit_domains  = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com"]
fraud_domains  = ["anonymous.com", "protonmail.com", None, "mail.com"]
p_email = [random.choice(legit_domains) if not f
           else random.choice(fraud_domains) for f in fraud_mask]
r_email = [random.choice(legit_domains) for _ in range(N)]

# Build DataFrame
df = pd.DataFrame({
    "TransactionID":  range(2987000, 2987000 + N),
    "isFraud":        fraud_mask.astype(int),
    "TransactionDT":  tx_dt,
    "TransactionAmt": tx_amt,
    "ProductCD":      product_cd,
    "card1":          card1,
    "card2":          card2,
    "card3":          card3,
    "card4":          card4,
    "card5":          card5,
    "card6":          card6,
    "addr1":          addr1,
    "addr2":          addr2,
    "dist1":          dist1,
    "dist2":          np.where(np.random.random(N) < 0.8, np.nan,
                               np.random.exponential(20, N).round(1)),
    "P_emaildomain":  p_email,
    "R_emaildomain":  r_email,
})

df.to_csv(args.output, index=False)
fraud_n = fraud_mask.sum()
print(f"[DONE] Generated {N:,} rows → {args.output}")
print(f"       Fraud: {fraud_n:,} ({100*fraud_n/N:.2f}%) | "
      f"Legit: {N-fraud_n:,}")
