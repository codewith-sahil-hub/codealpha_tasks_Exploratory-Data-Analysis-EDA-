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
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_style("whitegrid")
BASE_DIR = Path(__file__).resolve().parent
CHART_DIR = BASE_DIR / "charts"
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
df = pd.read_csv(BASE_DIR / "product_price_analysis.csv")

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

# Validate types and retain observed ratings for statistical tests.
required = {"title", "price_gbp", "rating_stars", "in_stock"}
absent = required - set(df.columns)
if absent:
    raise SystemExit(f"Missing required columns: {sorted(absent)}")
df_clean = df.drop_duplicates().copy()
for column in ["price_gbp", "rating_stars"]:
    df_clean[column] = pd.to_numeric(df_clean[column], errors="coerce")
    df_clean.loc[~np.isfinite(df_clean[column]), column] = np.nan
df_clean.loc[~df_clean["rating_stars"].isin([1, 2, 3, 4, 5]), "rating_stars"] = np.nan
df_clean["in_stock"] = df_clean["in_stock"].astype("string").str.strip().str.lower().map(
    {"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False}
)
invalid_prices = df_clean["price_gbp"].isna() | (df_clean["price_gbp"] <= 0)
print(f"Invalid/missing price rows removed: {invalid_prices.sum()}")
df_clean = df_clean.loc[~invalid_prices].copy()
if df_clean.empty:
    raise SystemExit("No valid positive prices remain; analysis stopped.")
# Complete cases avoid treating imputed ratings as measured observations.
observed_pairs = df_clean[["rating_stars", "price_gbp"]].dropna().copy()
missing_ratings = int(df_clean["rating_stars"].isna().sum())
median_rating = df_clean["rating_stars"].median() if df_clean["rating_stars"].notna().any() else np.nan
if pd.notna(median_rating):
    df_clean["rating_stars"] = df_clean["rating_stars"].fillna(median_rating)
df_clean["value_score"] = (df_clean["rating_stars"] / df_clean["price_gbp"]).round(3)
df_clean["price_bucket"] = pd.cut(df_clean["price_gbp"], [0, 15, 35, np.inf],
    labels=["budget", "mid-range", "premium"], right=False)
print(f"After cleaning: {len(df_clean)} rows; {missing_ratings} ratings missing before descriptive imputation.")
print("Rating charts use median-imputed values where possible; Pearson uses observed pairs only.")
print(f"Unknown stock statuses excluded from stock analysis: {df_clean['in_stock'].isna().sum()}")
print("\nStock availability by price tier (known statuses only):")
stock_known = df_clean.dropna(subset=["in_stock"])
if not stock_known.empty:
    stock_table = stock_known.groupby("price_bucket", observed=True)["in_stock"].agg(["count", "mean"])
    stock_table["out_of_stock_pct"] = (1 - stock_table.pop("mean")) * 100
    print(stock_table.round(2))
else:
    print("No known stock statuses available.")


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
if df_clean["rating_stars"].notna().any():
    sns.boxplot(data=df_clean, x="rating_stars", y="price_gbp")
else:
    plt.text(0.5, 0.5, "No valid ratings available", ha="center")
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

# H1: Pearson correlation using observed ratings only.
r = p_val = np.nan
print(f"H1: Rating vs. price ({len(observed_pairs)} observed pairs)")
if len(observed_pairs) >= 3 and (observed_pairs.nunique() > 1).all():
    r, p_val = stats.pearsonr(observed_pairs["rating_stars"], observed_pairs["price_gbp"])
    print(f"    Pearson r = {r:.3f}, p = {p_val:.3e}")
else:
    print("    Not tested: need at least 3 pairs and variation in both variables.")

# H2: Welch test using known stock status and valid prices.
in_stock_prices = stock_known.loc[stock_known["in_stock"] == True, "price_gbp"]
out_stock_prices = stock_known.loc[stock_known["in_stock"] == False, "price_gbp"]
p_val2 = np.nan
print("\nH2: In-stock vs. out-of-stock pricing (Welch test)")
print(f"    Group sizes: {len(in_stock_prices)} in stock, {len(out_stock_prices)} out of stock")
if len(in_stock_prices) >= 2 and len(out_stock_prices) >= 2 and (
    in_stock_prices.var() > 0 or out_stock_prices.var() > 0
):
    t_stat, p_val2 = stats.ttest_ind(in_stock_prices, out_stock_prices, equal_var=False)
    print(f"    Mean prices: £{in_stock_prices.mean():.2f} vs. £{out_stock_prices.mean():.2f}")
    print(f"    t = {t_stat:.3f}, p = {p_val2:.3e}")
else:
    print("    Not tested: need 2 records per group and nonzero variance.")


# ----------------------------------------------------------------------
# 5. Summary
# ----------------------------------------------------------------------
section("SUMMARY OF FINDINGS")
print(f"Original data: {dupes} duplicates; {df.isna().sum().sum()} missing cells.")
print(f"Cleaned data: {len(df_clean)} rows; {len(outliers)} IQR price outliers retained for review.")
print(f"Remaining missing cells: {df_clean.isna().sum().sum()}")
for label, probability in [("Rating-price correlation", p_val), ("Stock price difference", p_val2)]:
    result = "not tested" if not np.isfinite(probability) else ("significant" if probability < 0.05 else "not significant")
    print(f"{label}: {result} at alpha = 0.05.")
print("Sample data results illustrate the workflow; they are not evidence about real book markets.")
