"""
Task 2: Exploratory Data Analysis (EDA)
==========================================
Dataset: product_price_analysis.csv (book price/rating data, from Task 1)

This script follows the standard EDA workflow:
  1. Ask meaningful questions
  2. Explore structure (variables, types)
  3. Identify trends, patterns, anomalies
  4. Test hypotheses with statistics + visualization
  5. Detect data quality issues

Run with: python eda_analysis.py
Outputs: printed summary in the console + PNG charts in ./charts/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_style("whitegrid")
CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ----------------------------------------------------------------------
# 0. Questions we're asking BEFORE touching the data
# ----------------------------------------------------------------------
QUESTIONS = [
    "Is there a real relationship between rating and price, or is 'you get "
    "what you pay for' not actually true here?",
    "Are premium-priced items disproportionately out of stock?",
    "Where are the outliers, and are they data errors or real extreme values?",
    "Is the dataset clean enough to trust for downstream analysis "
    "(missing values, duplicates, wrong types)?",
]

section("STEP 1: Questions guiding this analysis")
for i, q in enumerate(QUESTIONS, 1):
    print(f"  {i}. {q}")


# ----------------------------------------------------------------------
# 1. Load + explore structure
# ----------------------------------------------------------------------
df = pd.read_csv("product_price_analysis.csv")

section("STEP 2: Data structure")
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
print("Dtypes:")
print(df.dtypes)
print("\nFirst rows:")
print(df.head())

section("Descriptive statistics (numeric columns)")
print(df.describe())


# ----------------------------------------------------------------------
# 2. Data quality check (do this before trusting any trend)
# ----------------------------------------------------------------------
section("STEP 3: Data quality issues")

missing = df.isnull().sum()
missing = missing[missing > 0]
print("Missing values by column:")
print(missing if not missing.empty else "  None found")

dupes = df.duplicated().sum()
print(f"\nExact duplicate rows: {dupes}")

# Clean for downstream analysis (report first, then handle)
df_clean = df.drop_duplicates().copy()
df_clean["rating_stars"] = df_clean["rating_stars"].fillna(df_clean["rating_stars"].median())
print(f"\nAfter cleaning: {df_clean.shape[0]} rows "
      f"(dropped {df.shape[0] - df_clean.shape[0]} duplicates, "
      f"imputed missing ratings with median)")


# ----------------------------------------------------------------------
# 3. Trends, patterns, anomalies
# ----------------------------------------------------------------------
section("STEP 4: Trends, patterns, anomalies")

print("Price bucket distribution:")
print(df_clean["price_bucket"].value_counts())

print("\nAverage price by rating:")
print(df_clean.groupby("rating_stars")["price_gbp"].mean().round(2))

# Outlier detection via IQR
q1, q3 = df_clean["price_gbp"].quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outliers = df_clean[(df_clean["price_gbp"] < lower) | (df_clean["price_gbp"] > upper)]
print(f"\nPrice outliers (IQR method): {len(outliers)} rows found "
      f"(outside £{lower:.2f}-£{upper:.2f})")

# --- Charts ---
plt.figure(figsize=(7, 4))
sns.histplot(df_clean["price_gbp"], bins=25, kde=True)
plt.title("Price Distribution")
plt.xlabel("Price (GBP)")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/price_distribution.png", dpi=120)
plt.close()

plt.figure(figsize=(7, 4))
sns.boxplot(data=df_clean, x="rating_stars", y="price_gbp")
plt.title("Price by Rating (outliers visible as dots)")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/price_by_rating.png", dpi=120)
plt.close()

plt.figure(figsize=(6, 5))
corr = df_clean[["price_gbp", "rating_stars", "value_score"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/correlation_heatmap.png", dpi=120)
plt.close()

print(f"\nCharts saved to ./{CHART_DIR}/")


# ----------------------------------------------------------------------
# 4. Hypothesis testing
# ----------------------------------------------------------------------
section("STEP 5: Hypothesis tests")

# H1: Rating and price are correlated
r, p_val = stats.pearsonr(df_clean["rating_stars"], df_clean["price_gbp"])
print(f"H1: Correlation between rating and price")
print(f"    Pearson r = {r:.3f}, p = {p_val:.4f}")
print(f"    {'Statistically significant' if p_val < 0.05 else 'Not statistically significant'} "
      f"correlation at alpha = 0.05")

# H2: In-stock items are priced differently than out-of-stock items
in_stock_prices = df_clean[df_clean["in_stock"] == True]["price_gbp"]
out_stock_prices = df_clean[df_clean["in_stock"] == False]["price_gbp"]
t_stat, p_val2 = stats.ttest_ind(in_stock_prices, out_stock_prices, equal_var=False)
print(f"\nH2: In-stock vs out-of-stock price difference (Welch's t-test)")
print(f"    In-stock mean: £{in_stock_prices.mean():.2f}  "
      f"Out-of-stock mean: £{out_stock_prices.mean():.2f}")
print(f"    t = {t_stat:.3f}, p = {p_val2:.4f}")
print(f"    {'Statistically significant' if p_val2 < 0.05 else 'No significant'} "
      f"difference at alpha = 0.05")


# ----------------------------------------------------------------------
# 5. Summary
# ----------------------------------------------------------------------
section("SUMMARY OF FINDINGS")
print(f"""
- Dataset had {dupes} duplicate rows and {missing.sum() if not missing.empty else 0}
  missing values before cleaning — both addressed above.
- {len(outliers)} price outliers detected via IQR; worth checking manually
  before removing (they may be legitimate premium items, not errors).
- Rating vs. price correlation: r = {r:.3f} ({'weak' if abs(r) < 0.3 else 'moderate' if abs(r) < 0.6 else 'strong'}).
- In-stock/out-of-stock price gap: {'significant' if p_val2 < 0.05 else 'not significant'}.
""")
