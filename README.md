# Assessment Alt — Graded Sports Card Price Prediction

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

1. **Exact match**: most recent prior sale of the same card key
2. **Subject + grade median**: fallback for unseen cards
3. **Subject median**: second fallback
4. **Global median**: final fallback

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
- The model and non-ML baselines use the same train/test split, enabling direct comparison
- The model overfits quickly due to the card-level price features creating a distribution shift between seen and unseen cards
- Prediction range is compressed ($20-$3K) vs actual ($1-$4M) — extreme outliers are unrecoverable

## Infrastructure

- **Storage**: S3 bucket `assessment-alt` with prefixed paths per task (e.g., `02_data_split/train.csv`)
- **Compute**: SageMaker notebooks (Python 3.10+)
- **Dependencies**: pandas, numpy, matplotlib, seaborn, xgboost, joblib

## Potential Improvements

- Variety-level statistics as features
- More granular time features (day of week, market momentum indicators)
- External data (player performance, awards)
- Ensemble approach: use exact match baseline when available, model otherwise
- Quantile regression for confidence intervals
- Address seen/unseen card distribution shift (separate models or indicator features)
