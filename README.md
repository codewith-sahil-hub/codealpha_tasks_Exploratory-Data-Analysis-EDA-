# CodeAlpha_ExploratoryDataAnalysis

## Task 2: Exploratory Data Analysis — Data Analytics Internship (CodeAlpha)

EDA on the book price/rating dataset produced in Task 1
(`CodeAlpha_WebScraping`), using **pandas, matplotlib, seaborn, and scipy**.

## Files

```
CodeAlpha_ExploratoryDataAnalysis/
├── eda_analysis.py              # Main EDA script (run this)
├── generate_sample_data.py      # Generates sample data for demo purposes
├── product_price_analysis.csv   # Sample dataset (swap for your real scraped data)
├── charts/                      # Output visualizations
│   ├── price_distribution.png
│   ├── price_by_rating.png
│   └── correlation_heatmap.png
└── requirements.txt
```

> **Note:** `product_price_analysis.csv` here is a generated sample matching
> the schema from Task 1 (title, price_gbp, price_bucket, rating_stars,
> in_stock, value_score), used because this dataset was built where live
> scraping wasn't possible. Replace it with your real Task 1 output —
> the script validates the required fields before analysis.

## EDA workflow followed

1. **Meaningful questions asked up front** (see top of `eda_analysis.py`) —
   e.g. does rating correlate with price, are premium items less often in
   stock, are outliers real or errors, is the data clean enough to trust.
2. **Structure exploration** — shape, dtypes, `.describe()`.
3. **Data quality check** — missing values, duplicate rows, before any
   conclusions are drawn.
4. **Trends & anomalies** — price distribution, average price by rating,
   IQR-based outlier detection.
5. **Hypothesis testing** — Pearson correlation (rating vs. price) and
   Welch's t-test (in-stock vs. out-of-stock pricing), both with p-values.
6. **Visualization** — histogram, boxplot, correlation heatmap saved to
   `charts/`.

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python eda_analysis.py
```

Console output includes structure summary, data quality report, and
hypothesis test results. Charts are saved to `charts/`.

## Data handling and interpretation

Requires Python 3.10 or newer. Run `python eda_analysis.py` after setup.
Input and charts are resolved relative to the script folder.
The supplied CSV is synthetic demo data, not a live scraped dataset.

Duplicates are removed. Invalid or missing prices and nonpositive prices
are excluded. Ratings outside integer stars 1–5 are treated as missing.
Median ratings are used for descriptive charts when observed ratings exist;
Pearson correlation uses only observed price/rating pairs. Value scores and
price tiers are recalculated. Unknown stock statuses remain unknown.
Stock availability is compared by price tier using known statuses only.
Statistical tests are skipped when samples or variation are insufficient.
P-values are shown in scientific notation; results are exploratory, do not
establish causation, and depend on the supplied data and test assumptions.
The value-score correlations partly reflect its definition (rating / price).

To regenerate the synthetic data, run `python generate_sample_data.py`.
This overwrites `product_price_analysis.csv` beside the script; back up real
data before running it. Analysis regenerates the three files in `charts/`.

## Notes

Completed as part of the CodeAlpha Data Analytics Internship —
Task 2: Exploratory Data Analysis.
