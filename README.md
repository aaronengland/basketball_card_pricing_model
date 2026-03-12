# Basketball Card Pricing Model

> **Goal**: Predict how much a graded basketball card will sell for, using historical sales data.

This project builds a machine learning pipeline to estimate the sale price of 2018 graded basketball cards. I walk through every step — from exploring the raw data, to engineering features, to training a model — and compare my ML approach against a simple "lookup the last sale" baseline.

> **Terminology note**: Throughout this document, **"txn"** is shorthand for **"transaction"** — a single recorded sale of a card at a specific price and date.

---

## Table of Contents

- [The Dataset](#the-dataset)
- [Step 1 — Exploratory Data Analysis](#step-1--exploratory-data-analysis)
- [Step 2 — Splitting the Data](#step-2--splitting-the-data)
- [Step 3 — Feature Engineering](#step-3--feature-engineering)
- [Step 4 — Non-ML Baseline](#step-4--non-ml-baseline)
- [Step 5 — XGBoost Model Training](#step-5--xgboost-model-training)
- [Step 6 — Model Evaluation](#step-6--model-evaluation)
- [Infrastructure](#infrastructure)
- [Discussion & Reflections](#discussion--reflections)

---

## The Dataset

The dataset contains **100,000 sales records** of graded basketball cards from 2018, spanning **May 2018 through February 2022**.

Each record captures:

| Field | What It Represents |
|---|---|
| `year` | The year the card was produced (always 2018) |
| `subject` | The player on the card (e.g., LeBron James) |
| `brand` | The card manufacturer (e.g., Panini Prizm) |
| `variety` | The specific variant (e.g., "Silver", or null for base cards) |
| `card_number` | The number printed on the card |
| `date` | When the sale happened |
| `price` | What it sold for (the prediction target) |
| `grade` | The condition rating (1–10 scale) |
| `grading_company` | Who graded it (PSA, BGS, SGC, etc.) |

**What makes two cards "the same"?** Cards are considered identical (fungible) when they share the same year, subject, brand, variety, card number, grade, and grading company. This combination is called the **card key** — think of it as a unique fingerprint for a specific card type.

**Source**: Stored in S3 at `s3://assessment-alt/00_data_collection/transaction_data.csv`

---

## Step 1 — Exploratory Data Analysis

> **Notebook**: [`01_eda/notebook.ipynb`](01_eda/notebook.ipynb)

Before building anything, I took a close look at the data to understand what I was working with. Here's what stood out:

### Prices are wildly skewed

Most cards sell for under $100, but a few sell for hundreds of thousands — even millions. The typical (median) sale is **$95**, the average is **$460**, and the highest sale in the dataset is **$4 million**. Because of this extreme skew, I apply a **log transformation** to prices before modeling — this brings the distribution closer to normal and prevents outliers from dominating the model.

<p align="center">
  <img src="01_eda/output/target_distribution.png" alt="Price distribution before and after log transform" width="700"/>
</p>

### Grade 10 cards dominate the market

Over half (51%) of all transactions involve **Grade 10** (gem mint) cards, and another 36% are **Grade 9**. Lower grades are relatively rare in this dataset, meaning the market is heavily concentrated around top-condition cards.

<p align="center">
  <img src="01_eda/output/grade_distribution.png" alt="Grade distribution" width="700"/>
</p>

### PSA is the go-to grading company

**87% of cards** were graded by PSA (Professional Sports Authenticator), making it the overwhelming standard. BGS, SGC, and others make up the remaining 13%.

<p align="center">
  <img src="01_eda/output/grading_company_distribution.png" alt="Grading company distribution" width="700"/>
</p>

### Prices vary significantly by grade and grader

Higher grades command higher prices, as expected — but the grading company also matters. PSA cards tend to carry a premium.

<p align="center">
  <img src="01_eda/output/price_by_grade.png" alt="Price by grade" width="700"/>
</p>

<p align="center">
  <img src="01_eda/output/price_by_grading_company.png" alt="Price by grading company" width="700"/>
</p>

### Transaction volume exploded after COVID

Card sales surged dramatically starting in 2020, likely driven by the COVID-era collectibles boom. This matters for modeling because it means the market environment changed significantly over the dataset's timeframe.

<p align="center">
  <img src="01_eda/output/transactions_over_time.png" alt="Transaction volume over time" width="700"/>
</p>

### Top players and brands

A handful of players and brands dominate transaction volume. This concentration means the model will have much more data for popular cards than for niche ones.

<p align="center">
  <img src="01_eda/output/top_subjects.png" alt="Top subjects (players) by transaction count" width="700"/>
</p>

<p align="center">
  <img src="01_eda/output/top_brands.png" alt="Top brands by transaction count" width="700"/>
</p>

### Price trends over time for top players

Some players' card values moved dramatically over the dataset's timeframe, reflecting real-world events, hype cycles, and market dynamics.

<p align="center">
  <img src="01_eda/output/price_over_time_top_subjects.png" alt="Price trends over time for top subjects" width="700"/>
</p>

### Data quality issues

- **~45% of the `variety` field is null** — these are base cards with no special variant. I fill these with "base" during preprocessing.
- **Case inconsistencies**: The `subject` and `brand` fields have duplicates caused by inconsistent capitalization (e.g., "LeBron James" vs "LEBRON JAMES") — 231 duplicate subjects and 139 duplicate brands. I normalize these during preprocessing.

<p align="center">
  <img src="01_eda/output/propna.png" alt="Missing value proportions" width="700"/>
</p>

<p align="center">
  <img src="01_eda/output/variety_analysis.png" alt="Variety analysis" width="700"/>
</p>

---

## Step 2 — Splitting the Data

> **Notebook**: [`02_data_split/notebook.ipynb`](02_data_split/notebook.ipynb)

I split the data **by date** (not randomly) to simulate real-world conditions — the model only sees past sales and must predict future ones. This prevents "lookahead bias," where the model accidentally learns from data it wouldn't have in production.

Here's how the splits break down:

- **Training set (62%)** — May 2018 through May 2021 — 62,407 transactions. This is what the model learns from.
- **Validation set (22%)** — June through October 2021 — 21,535 transactions. Used to tune the model and decide when to stop training.
- **Test set (16%)** — November 2021 through February 2022 — 16,058 transactions. The final, untouched evaluation set.

Grade and grading company distributions stay fairly consistent across all three splits. However, prices shift over time due to evolving market conditions — which is exactly the challenge a real-world model would face.

<p align="center">
  <img src="02_data_split/output/split_distribution.png" alt="Split distribution" width="700"/>
</p>

<p align="center">
  <img src="02_data_split/output/date_distribution.png" alt="Date distribution across splits" width="700"/>
</p>

---

## Step 3 — Feature Engineering

> **Notebook**: [`03_preprocessing/notebook.ipynb`](03_preprocessing/notebook.ipynb)

Raw fields like player names and brand strings can't be fed directly into a model. I built a preprocessing pipeline with **8 transformers** that convert the raw data into 13 numeric features plus a log-transformed price target.

Here's what each transformer does, in plain language:

| Transformer | What It Does |
|---|---|
| **TextNormalizer** | Cleans up text — lowercases everything, strips whitespace, fills missing variety with "base" |
| **CardKeyFeatures** | For each unique card, calculates how many times it sold in training and its median/mean price. These give the model a sense of "what has this exact card been going for?" |
| **SubjectFeatures** | Same idea but at the player level — how often does this player's cards sell, and at what median price? |
| **BrandFeatures** | Same idea at the brand level — how popular and expensive is this brand? |
| **GradeFeatures** | Converts grade to a number, flags whether it's PSA-graded, and flags "gem mint" (grade 10) |
| **TimeFeatures** | Extracts the year/month of sale and how many days have passed since the first transaction in the dataset |
| **TargetTransform** | Applies log(price + 1) to the sale price, taming the extreme skew I found in EDA |
| **PrepareFeatures** | Selects the final 13 feature columns for modeling |

**Important detail**: The "stateful" transformers (CardKey, Subject, Brand) are computed **only from training data**. When I process validation or test data, I look up those same training-derived statistics — I never peek at future information. The fitted pipeline is saved as a `.joblib` file so it can be reused consistently.

---

## Step 4 — Non-ML Baseline

> **Notebook**: [`04_price_estimator/notebook.ipynb`](04_price_estimator/notebook.ipynb)

Before reaching for machine learning, I built a simple **lookup-based baseline** to see how far common sense gets you. The approach is a cascade — try the most specific method first, and fall back to broader estimates:

1. **Exact match** — Find the most recent prior sale of this exact card key. If it exists, use that price. This works for **78.6%** of test cards.
2. **Subject + grade median** — For cards I haven't seen before, take the median price of all cards with the same player and grade. Covers **99.2%**.
3. **Subject median** — If there's still no match, use the median price for that player across all grades. Covers **99.9%**.
4. **Global median** — Last resort: use the overall median price. Covers **100%**.

### How did it perform?

- **Exact match alone** was remarkably strong: it achieved an R-squared of **0.64** and a median error of just **$39** — but only covers 78.6% of cards.
- **The combined cascade** (using all fallbacks) covers every card but is weaker overall: median error of **$40.49** and R-squared near zero. The fallback methods dilute the accuracy of the exact-match approach.

The takeaway: for cards with prior sales history, **simply looking up the last sale price is hard to beat**.

<p align="center">
  <img src="04_price_estimator/output/actual_vs_predicted.png" alt="Baseline actual vs predicted" width="700"/>
</p>

<p align="center">
  <img src="04_price_estimator/output/coverage.png" alt="Baseline coverage by method" width="700"/>
</p>

---

## Step 5 — XGBoost Model Training

> **Notebook**: [`05_model/notebook.ipynb`](05_model/notebook.ipynb)

I trained an **XGBoost** gradient-boosted tree model on the 13 engineered features, predicting the log-transformed price.

### Model configuration

| Parameter | Value | Why |
|---|---|---|
| Number of rounds | 1,000 (max) | Upper bound on boosting iterations; early stopping decides the actual count |
| Learning rate | 0.05 | Slow learning for better generalization |
| Max tree depth | 6 | Moderate complexity per tree |
| Min child weight | 30 | Prevents trees from splitting on tiny groups |
| Subsample | 80% | Each tree sees a random 80% of rows (reduces overfitting) |
| Column sample | 80% | Each tree sees a random 80% of features |
| Early stopping | 100 rounds | Stop if validation error hasn't improved in 100 rounds |

### What happened

The model **stopped early at just 18 boosting rounds** — a sign of rapid overfitting. Here's the error progression (RMSE on the log-price scale):

- **Training**: 0.62
- **Validation**: 1.06
- **Test**: 1.32

The gap between training and validation/test error tells us the model memorized the training data rather than learning generalizable patterns. The likely culprit: **card-level price features** (median/mean price from training) act as near-direct answers for cards the model has seen before, but are zero for unseen cards — creating a split personality problem.

> **Scope note**: This iteration skipped hyperparameter tuning and feature selection (RFE) for time reasons. Both are natural next steps.

---

## Step 6 — Model Evaluation

> **Notebook**: [`06_model_eval/notebook.ipynb`](06_model_eval/notebook.ipynb)

I evaluated the XGBoost model on the held-out test set and compared it to my non-ML baseline.

### Overall test set performance

- **Mean Absolute Error (MAE)**: $586.69 — on average, predictions are off by ~$587
- **Median Absolute Error**: $57.75 — half of predictions are within ~$58 (the median is more representative given the skewed prices)
- **MAPE**: 192.47% — percentage error is high because small-dollar cards with even modest dollar errors produce huge percentage errors
- **R-squared**: -0.0001 — the model explains essentially none of the variance (roughly equivalent to predicting the mean for everything)

### Seen vs. unseen cards

The model behaves very differently depending on whether it has seen a card before:

- **Seen cards (78.6% of test set)**: Median error of $61.64, R-squared of 0.16 — decent but not great
- **Unseen cards (21.4%)**: Median error of $17.72, R-squared of -0.0008 — low error in dollars (these tend to be cheaper cards), but no predictive power

### The honest takeaway

**The simple lookup baseline beats the ML model on every metric.** The baseline achieved lower MAE ($571 vs $587), lower median error ($40 vs $58), and marginally higher R-squared. For fungible items with transaction history, "just look up the last sale" turns out to be a very strong strategy that's difficult for a model to surpass.

The model also compresses its prediction range to roughly $20–$3,000, while actual prices span $1 to $4 million. Extreme outliers are simply unrecoverable.

<p align="center">
  <img src="06_model_eval/output/actual_vs_predicted.png" alt="Model actual vs predicted" width="700"/>
</p>

<p align="center">
  <img src="06_model_eval/output/residual_analysis.png" alt="Residual analysis" width="700"/>
</p>

<p align="center">
  <img src="06_model_eval/output/error_by_grade.png" alt="Error breakdown by grade" width="700"/>
</p>

<p align="center">
  <img src="06_model_eval/output/error_by_price_range.png" alt="Error breakdown by price range" width="700"/>
</p>

<p align="center">
  <img src="06_model_eval/output/feature_importance.png" alt="Feature importance" width="700"/>
</p>

---

## Infrastructure

| Component | Details |
|---|---|
| **Storage** | S3 bucket `assessment-alt`, organized by pipeline step (e.g., `02_data_split/train.csv`) |
| **Compute** | AWS SageMaker notebooks, Python 3.10+ |
| **Key libraries** | pandas, numpy, matplotlib, seaborn, xgboost, joblib |

---

## Discussion & Reflections

### How would you generate a confidence interval for your prediction?

Two practical approaches:

1. **Quantile regression** — Train XGBoost with `reg:quantileerror` to predict the 10th and 90th percentiles separately, giving you a prediction interval directly from the model.
2. **Empirical distribution** — For seen cards, compute percentile ranges from that card's historical sales. For unseen cards, fall back to the player+grade distribution. The non-ML baseline lends itself naturally to this since it's already built on group-level aggregations.

### What are the shortcomings of your approach?

- The non-ML baseline **outperforms the XGBoost model on every metric**, meaning the model isn't adding value beyond simple lookups for this particular dataset.
- **21.4% of test cards are completely new** — the model has no card-specific pricing signal for these and must rely solely on player/brand/grade features.
- **Extreme prices are unrecoverable** — the log transform and model architecture cap predictions around $3K, while real sales reach $4M.
- **No hyperparameter tuning or feature selection** was performed due to time constraints.
- The **temporal split forces the model to generalize across different market regimes** (pre-COVID calm vs. post-COVID boom), which is inherently difficult.

### What features would you engineer with more time?

- **Variety-level price stats** (paralleling the card/subject/brand statistics I already built)
- **Market momentum** — rolling median price over the last 30/60/90 days for a card or player, capturing trends
- **Price velocity** — how fast a card's price is changing over time
- **Seen/unseen indicator** — a binary flag so the model can learn distinct strategies for known vs. unknown cards
- **External data** — player stats, All-Star/MVP selections, injury status, rookie vs. veteran status
- **Seasonality features** — day of week, holidays, NBA playoff timing

### Would more data help? (10x? 100x?)

**10x would likely help significantly.** The biggest bottleneck is coverage — only 78.6% of test cards appear in training. More data means more cards have sales history, improving both the model's card-level features and the exact-match baseline. It would also produce better estimates for niche players and brands in the long tail.

**100x would help further, but with diminishing returns.** Eventually you'd saturate coverage for 2018 basketball cards, and the remaining prediction error would come from genuine price volatility — market shifts, subjectivity within a grade, auction dynamics — that no amount of historical data can eliminate. The core challenge is that card prices are **nonstationary**: more data from 2018–2019 doesn't necessarily help predict 2022 prices.

### How did you prevent lookahead bias?

Lookahead bias happens when a model accidentally uses information from the future during training — for example, if price statistics included sales that hadn't happened yet at the time of prediction. To prevent this, I used a strict **out-of-time split**: training data ends in May 2021, validation covers June–October 2021, and the test set starts in November 2021. All stateful preprocessing (card/subject/brand statistics) is computed exclusively from training data. The fitted `PreprocessingModel` stores these transformers and applies transform-only to validation and test sets — no future information leaks backward. The price estimator baselines follow the same rule, only looking up prices from the training period.

### How did you address overfitting?

Overfitting is when a model memorizes the training data too closely — it performs well on data it has already seen but poorly on new, unseen data. To combat this, I used multiple mechanisms:

- **Early stopping** on validation RMSE (triggered at round 18)
- **Regularization** — min_child_weight=30, gamma=0.1, L2 lambda=1.0, L1 alpha=0.1
- **Subsampling** — 80% of rows and 80% of columns per tree

Despite all of this, the model overfits quickly because the card-level price features create a **bimodal problem**: for seen cards, the historical price essentially gives the answer away; for unseen cards, those features are zero and provide no signal. With more time, next steps would include feature selection (RFE), hyperparameter tuning, and potentially **separate models or an ensemble** for seen vs. unseen cards.
