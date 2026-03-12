# Basketball Card Pricing Model

A machine learning pipeline for predicting the sale price of graded basketball cards using historical transaction data. The project is structured as a series of notebooks, each handling a discrete step in the pipeline.

## Data

- **Source**: 100,000 transactions of 2018 graded basketball cards, stored in S3 (`s3://assessment-alt/00_data_collection/transaction_data.csv`)
- **Date range**: May 2018 – Feb 2022
- **Features**: year, subject (player), brand, variety, card_number, date, price, grade, grading_company
- **Card identity**: Cards are fungible when (year, subject, brand, variety, card_number, grade, grading_company) match — this tuple forms the card key

## Pipeline

### 01_eda — Exploratory Data Analysis

Profiles the raw transaction data to understand distributions, missing values, and data quality issues.

Key findings:
- Price is heavily right-skewed (median $95, mean $460, max $4M) — log transform is essential
- Grade 10 dominates at ~51% of transactions, grade 9 at ~36%
- PSA is the dominant grader (87%)
- ~45% of variety is null (base cards)
- Subject and brand have significant case inconsistencies (231 and 139 duplicates respectively)
- Transaction volume spiked post-COVID (2020+)

### 02_data_split — Train/Valid/Test Split

Splits data chronologically to prevent lookahead bias:

| Split | Date Range | Rows | % |
|---|---|---|---|
| Train | 2018-05-16 to 2021-05-31 | 62,407 | 62.4% |
| Valid | 2021-06-01 to 2021-10-31 | 21,535 | 21.5% |
| Test | 2021-11-01 to 2022-02-17 | 16,058 | 16.1% |

Grade and grading company distributions are relatively stable across splits. Price distributions shift over time due to market trends.

### 03_preprocessing — Feature Engineering

Builds a preprocessing pipeline with 8 transformers, each fit individually on training data then collected into a `PreprocessingModel` for consistent transform-only application to all splits.

**Transformers**:

| Transformer | Type | Description |
|---|---|---|
| TextNormalizer | Stateless | Lowercases/strips text fields, fills null variety with "base" |
| CardKeyFeatures | Stateful | Card-level stats (txn count, median/mean price) from training |
| SubjectFeatures | Stateful | Subject-level stats (txn count, median price) from training |
| BrandFeatures | Stateful | Brand-level stats (txn count, median price) from training |
| GradeFeatures | Stateless | Numeric grade, is_psa flag, is_gem_mint flag |
| TimeFeatures | Stateless | Year/month sold, days since first transaction |
| TargetTransform | Stateless | Log(price + 1) target |
| PrepareFeatures | Stateless | Selects final 13 feature columns |

**Output**: 13 numeric features + 1 log-transformed target per row.

The fit `PreprocessingModel` is saved as a joblib artifact for reproducible inference.

### 04_price_estimator — Non-ML Baseline

Establishes a non-ML price estimation baseline using a cascade of lookup strategies. Uses the same train/test splits from 02_data_split for apples-to-apples comparison with the model.

1. **Exact match**: most recent prior sale of the same card key (78.6% coverage)
2. **Subject + grade median**: fallback for unseen cards (99.2% coverage)
3. **Subject median**: second fallback (99.9% coverage)
4. **Global median**: final fallback

| Method | N | MAE | Median AE | R² |
|---|---|---|---|---|
| Exact Match Only | 12,625 | $158.72 | $39.00 | 0.6375 |
| Subject+Grade Median | 15,929 | $620.53 | $65.00 | -0.0001 |
| Combined (all fallbacks) | 16,058 | $571.43 | $40.49 | 0.0007 |

### 05_model — XGBoost Training

Trains an XGBoost regressor (`reg:squarederror`) on the log-transformed price target with early stopping on validation RMSE.

**Parameters**: learning_rate=0.05, max_depth=6, min_child_weight=30, subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=100.

**Scope note**: For the sake of time and simplicity, this iteration does not include hyperparameter tuning (e.g., grid search, Bayesian optimization) or recursive feature elimination (RFE). Both are natural next steps for a production model. The card-level price features (median/mean) are retained despite contributing to rapid overfitting — they encode genuinely useful historical price signal for seen cards (the same signal the non-ML exact-match baseline relies on), and removing them would degrade seen-card predictions without meaningfully helping unseen cards.

**Results**: The model early-stopped at 18 boosting rounds. Train/Valid/Test RMSE (log scale): 0.62 / 1.06 / 1.32. The rapid overfitting is likely driven by card-level price features that act as near-direct targets for seen cards but are 0 for unseen cards.

### 06_model_eval — Model Evaluation

Evaluates the XGBoost model on the test set across multiple dimensions:

**Overall metrics** (actual price scale): MAE $586.69, MAPE 192.47%, Median AE $57.75, R² -0.0001

**Seen vs unseen cards**:
- Seen (78.6%): Median AE $61.64, R² 0.1557
- Unseen (21.4%): Median AE $17.72, R² -0.0008

**Key observations**:
- The combined non-ML baseline outperforms the XGBoost model on every metric — lower MAE ($571 vs $587), lower Median AE ($40 vs $58), and higher R² (0.0007 vs -0.0001). The exact-match baseline alone (R² 0.6375) is far stronger for seen cards, suggesting the simple lookup-based approach is hard to beat for fungible card pricing.
- The model overfits quickly (18 rounds) due to the card-level price features creating a distribution shift between seen and unseen cards
- Prediction range is compressed ($20-$3K) vs actual ($1-$4M) — extreme outliers are unrecoverable

## Infrastructure

- **Storage**: S3 bucket `assessment-alt` with prefixed paths per task (e.g., `02_data_split/train.csv`)
- **Compute**: SageMaker notebooks (Python 3.10+)
- **Dependencies**: pandas, numpy, matplotlib, seaborn, xgboost, joblib

## Discussion

**How would you generate a confidence interval for your prediction?**

Two approaches: (1) Use XGBoost's quantile regression (`reg:quantileerror`) to train separate models for the 10th and 90th percentiles, producing a prediction interval directly. (2) Leverage the card-level transaction history — for seen cards, compute empirical percentiles from the training price distribution for that card key. For unseen cards, fall back to the subject+grade distribution. The non-ML baseline has a natural advantage here since it's already built on group-level aggregations that extend readily to distributional estimates.

**What are the shortcomings of your approach?**

- The non-ML baseline outperforms the XGBoost model on every metric, indicating the model isn't adding value beyond simple lookups for this dataset
- 21.4% of test cards are unseen in training — the model has no card-specific signal for these and relies entirely on subject/brand/grade features
- Extreme prices (e.g., the $4M outlier) are unrecoverable — the log transform compresses the scale and the model's max prediction is ~$3K
- No hyperparameter tuning or recursive feature elimination was performed
- The temporal split means the model must generalize across market regimes (post-COVID boom/bust), which is inherently difficult

**What features would you engineer if you had more time?**

- Variety-level price statistics (paralleling the card/subject/brand stats)
- Market momentum features — rolling median price over the last 30/60/90 days for a card or subject, to capture trends
- Price velocity — rate of price change over time for a given card key
- Seen/unseen indicator — a binary flag for whether the card appeared in training, so the model can learn different strategies for each
- External data — player stats, All-Star/MVP selections, injury status, rookie year vs veteran
- Day-of-week and seasonality features (holiday effects, NBA playoff timing)

**Would your model improve with 10x the data? 100x?**

10x would likely help — the main bottleneck is coverage. Only 78.6% of test cards are seen in training. More data means more cards have transaction history, improving both the model's card-level features and the exact-match baseline. It would also yield better estimates for subject/brand stats in the long tail. 100x would help further but with diminishing returns — at some point you'd saturate coverage for 2018 basketball cards and the remaining error would come from genuine price volatility (market shifts, condition subjectivity within a grade, auction dynamics) that no amount of historical data resolves. The fundamental challenge is that card prices are nonstationary — more data from 2018-2019 doesn't necessarily help predict 2022 prices.

**How did/would you deal with lookahead bias?**

We used an out-of-time split — train is strictly before June 2021, validation is June–October 2021, test is November 2021 onward. All stateful preprocessing (card/subject/brand stats) is fit only on training data. The `PreprocessingModel` stores the fit transformers and applies transform-only to validation and test, so no future information leaks backward. The price estimator baselines also only look up prices from the training period.

**How did/would you deal with overfitting?**

We used several mechanisms: early stopping on validation RMSE (triggered at 18 rounds), regularization parameters (min_child_weight=30, gamma=0.1, lambda=1.0, alpha=0.1), and subsampling (80% row and column sampling per tree). Despite this, the model overfits quickly because the card-level price features create a bimodal problem — for seen cards the answer is essentially given, for unseen cards there is no signal. With more time, next steps would include: RFE to find a more generalizable feature set, hyperparameter tuning to optimize the bias-variance tradeoff, and potentially separate models or an ensemble strategy for seen vs unseen cards.
