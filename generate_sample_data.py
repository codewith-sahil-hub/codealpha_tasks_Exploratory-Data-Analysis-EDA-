"""
Generates a realistic SAMPLE dataset matching the schema produced by
custom_dataset_example.py from Task 1 (book price/rating data).

This exists ONLY so the EDA script has something to run against in an
offline environment. Replace product_price_analysis.csv with your real
scraped output before submitting — the EDA script works identically
on either.
"""
from pathlib import Path
import numpy as np
import pandas as pd

np.random.seed(42)
N = 300

titles = [f"Book Title {i}" for i in range(1, N + 1)]
rating_stars = np.random.choice([1, 2, 3, 4, 5], size=N, p=[0.08, 0.15, 0.30, 0.30, 0.17])

# Price correlates loosely with rating, with noise + some deliberate outliers
base_price = 10 + rating_stars * 4 + np.random.normal(0, 6, N)
base_price = np.clip(base_price, 2, None)
# inject a few outliers
outlier_idx = np.random.choice(N, 6, replace=False)
base_price[outlier_idx] = base_price[outlier_idx] * np.random.uniform(3, 5, 6)
price_gbp = np.round(base_price, 2)

def bucket(p):
    if p < 15:
        return "budget"
    elif p < 35:
        return "mid-range"
    return "premium"

price_bucket = [bucket(p) for p in price_gbp]
in_stock = np.random.choice([True, False], size=N, p=[0.85, 0.15])
value_score = np.round(rating_stars / price_gbp, 3)

df = pd.DataFrame({
    "title": titles,
    "price_gbp": price_gbp,
    "price_bucket": price_bucket,
    "rating_stars": rating_stars,
    "in_stock": in_stock,
    "value_score": value_score,
})

# Inject realistic data issues for the EDA to detect
missing_idx = np.random.choice(N, 8, replace=False)
df.loc[missing_idx, "rating_stars"] = np.nan
dup_rows = df.sample(4, random_state=1)
df = pd.concat([df, dup_rows], ignore_index=True)

df.to_csv(Path(__file__).resolve().parent / "product_price_analysis.csv", index=False)
print(f"Sample dataset generated: {df.shape[0]} rows, {df.shape[1]} columns")

