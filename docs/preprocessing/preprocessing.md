# Phase 5 — Data Preprocessing

## 1. Purpose

This document records the finalized data preprocessing strategy for the Credit Risk Platform.

The objective of preprocessing is to convert the raw Lending Club accepted-loan dataset into a clean, leakage-safe, model-ready representation while preserving information that may be useful for later feature engineering.

Preprocessing is deliberately separated from Exploratory Data Analysis (EDA) and Feature Engineering.

The responsibilities are therefore:

* **EDA** determines what the dataset contains and what decisions are appropriate.
* **Preprocessing** applies the finalized data-quality and representation decisions.
* **Feature Engineering** creates new representations and derived variables from the retained information.
* **Modeling** consumes the resulting processed datasets.

The production preprocessing workflow does **not** depend on the EDA notebook or the EDA Parquet artifact.

---

## 2. Production Input

The canonical input to preprocessing is the original accepted-loans CSV:

```text
data/raw/extracted/accepted_2007_to_2018Q4.csv
```

The preprocessing implementation loads this file through:

```text
DatasetLoader
    ↓
pd.read_csv(...)
```

The EDA-generated dataset:

```text
data/interim/eda_dataset.parquet
```

is intentionally **not** used by the production preprocessing pipeline.

### Why?

The EDA dataset was an intermediate artifact created specifically to support exploratory analysis. It should not become an implicit dependency of the actual ML pipeline.

The final architecture is therefore:

```text
Original Raw CSV
      │
      ▼
DatasetLoader
      │
      ▼
Target Construction
      │
      ▼
Modeling Population
      │
      ▼
Train / Validation / Test Split
      │
      ▼
Preprocessing
      │
      ├── Train
      ├── Validation
      └── Test
```

This also means that if the raw Lending Club dataset is replaced by a newer version, preprocessing can be executed from the source data without requiring a notebook-generated intermediate dataset.

---

# 3. Modeling Population Construction

## 3.1 Target Definition

The raw `loan_status` column contains both finalized and ongoing loan outcomes.

For supervised default prediction, a loan must have an observed final outcome.

The `TargetBuilder` therefore creates the binary `default` target.

### Default

The following statuses are treated as defaults:

```text
Charged Off
Default
Does not meet the credit policy. Status:Charged Off
```

These receive:

```text
default = 1
```

### Non-default

The following statuses are treated as non-defaults:

```text
Fully Paid
Does not meet the credit policy. Status:Fully Paid
```

These receive:

```text
default = 0
```

### Excluded outcomes

Loans whose eventual outcome has not been observed are excluded from the modeling population.

This prevents the model from treating an ongoing loan as either a successful repayment or a default when the final outcome is not yet known.

---

## 3.2 Modeling Population Result

From the EDA input population:

```text
EDA input rows:              2,260,701
Observed-outcome rows:       1,348,099
Excluded rows:                 912,602
Observed-outcome percentage:     59.63%
Default rate:                    19.98%
```

The resulting modeling population therefore contains:

```text
1,348,099 loans
```

with a default rate of approximately:

```text
19.98%
```

This population is the basis for all subsequent train/validation/test splits.

---

# 4. Feature Decisions from EDA

Preprocessing does not independently decide whether a feature is useful.

The feature-level decisions were established during Data Understanding and EDA.

The finalized registry classified the features into appropriate processing routes.

The important principle was:

> A feature being statistically interesting does not automatically mean it should be used in its current representation.

Similarly:

> A feature having missing values does not automatically mean it should be dropped.

The preprocessing stage therefore applies the decisions already established during analysis.

---

# 5. Feature Routing

The approved feature population contained:

```text
85 KEEP
8 REVIEW/DROP
1 TARGET
```

The 8 reviewed features were intentionally excluded from the Phase 5 model matrix:

```text
funded_amnt
funded_amnt_inv
int_rate
installment
grade
sub_grade
issue_d
initial_list_status
```

The target was:

```text
loan_status
```

and was converted into:

```text
default
```

The remaining approved features were routed according to their preprocessing requirements.

---

# 6. Phase 5 vs Phase 6 Boundary

One of the most important design decisions in this phase was separating **preprocessing** from **feature engineering**.

Not every retained feature needs to be transformed immediately.

Four approved features were deliberately deferred:

### High-cardinality features

```text
emp_title
title
zip_code
```

### Datetime feature

```text
earliest_cr_line
```

These features are retained in the modeling source data but are not passed through the Phase 5 transformation matrix.

Therefore:

```text
85 approved features
│
├── 81 → Phase 5 preprocessing
│
└── 4 → deferred to Phase 6
      ├── emp_title
      ├── title
      ├── zip_code
      └── earliest_cr_line
```

This distinction is intentional.

### Why defer these features?

`emp_title`, `title`, and `zip_code` require decisions about:

* cardinality reduction
* grouping
* rare-category treatment
* potential target encoding
* geographic representation
* text normalization

Those are feature-engineering decisions rather than simple data-cleaning operations.

Likewise, `earliest_cr_line` is a date representation. Converting it into a useful credit-history variable such as account age is a feature-engineering decision.

Therefore, forcing these features into the Phase 5 pipeline would blur the boundary between preprocessing and feature engineering.

---

# 7. Numerical Feature Preprocessing

Numerical features were divided into groups because missingness has different meanings across the Lending Club dataset.

A single generic "fill every numeric column with the median" strategy was deliberately avoided.

---

## 7.1 Standard Numerical Features

For ordinary numerical features, missing values are handled using training-set statistics.

The transformation also creates a missingness indicator:

```text
feature
feature__missing
```

For example:

```text
annual_inc
annual_inc__missing
```

### Why preserve missingness?

A missing value is not always random.

For example, a missing financial or credit attribute may indicate that:

* the attribute was not applicable,
* the information was unavailable,
* a particular application type did not provide it,
* or the data collection process behaved differently for that observation.

Replacing the value alone would erase that information.

The missingness indicator preserves the distinction between:

```text
observed value
```

and:

```text
originally missing value
```

---

# 8. Event-History Features

Several Lending Club variables represent elapsed time since a credit event.

Examples include:

```text
mths_since_last_delinq
mths_since_last_record
mths_since_recent_bc
mths_since_recent_inq
mths_since_recent_revol_delinq
```

For these features, a missing value can have a fundamentally different interpretation from an ordinary missing measurement.

For example:

```text
mths_since_last_delinq = NaN
```

may mean that the borrower has no recorded delinquency rather than that the value was simply forgotten.

Therefore these features use a sentinel-style treatment together with an explicit missingness indicator.

Conceptually:

```text
observed event
    ↓
actual elapsed-time value

no observed event
    ↓
sentinel representation
    +
missing indicator
```

This preserves the distinction between:

* an observed event with an elapsed time;
* and absence of the underlying event.

The sentinel is learned from the training population rather than arbitrarily hard-coded.

---

# 9. Structural Missingness

Some groups of Lending Club features exhibit **structural missingness**.

The clearest example is the group of joint-application variables.

For example:

```text
annual_inc_joint
dti_joint
verification_status_joint
```

are primarily relevant to joint applications.

Analysis showed:

```text
application_type = Individual
    → joint numerical fields are structurally unavailable

application_type = Joint App
    → joint numerical fields may be populated
```

This means that the missingness is not necessarily a data-quality problem.

It is partially determined by the application structure itself.

Therefore preprocessing preserves:

1. the feature-level missingness indicators; and
2. a structural-block missingness indicator.

The structural indicator captures the higher-level condition:

```text
entire structural feature block is missing
```

rather than treating every missing value as an independent random event.

---

# 10. Joint Application Features

The joint-application numerical features are processed separately:

```text
annual_inc_joint
dti_joint
```

The joint categorical feature is:

```text
verification_status_joint
```

Their separate treatment reflects the fact that their missingness is strongly associated with:

```text
application_type
```

Rather than dropping these features simply because they have very high overall missingness, the missingness mechanism is preserved.

This is an important example of why:

> High missing percentage alone is not sufficient justification for dropping a feature.

---

# 11. Categorical Features

Categorical variables are handled using two stages.

### Stage 1 — Missing category

Missing categorical values are explicitly represented as:

```text
__MISSING__
```

rather than silently removing the observations or treating the missing value as an arbitrary existing category.

### Stage 2 — One-hot encoding

Categorical variables are transformed using:

```text
OneHotEncoder(
    handle_unknown="ignore"
)
```

The `handle_unknown="ignore"` setting is important for future inference.

A category appearing in validation, test, or future production data that was not present during training will not cause the preprocessing pipeline to fail.

Instead, it receives an all-zero representation for that categorical encoder.

---

# 12. Train / Validation / Test Split

The modeling population is divided using a:

```text
70% train
15% validation
15% test
```

split.

The split is stratified on:

```text
default
```

This preserves the approximate default rate across the three populations.

Final result:

| Split      |    Rows | Default Rate |
| ---------- | ------: | -----------: |
| Train      | 943,669 |       19.98% |
| Validation | 202,215 |       19.98% |
| Test       | 202,215 |       19.98% |

The split is reproducible using:

```text
random_state = 42
```

---

# 13. Leakage Prevention During Preprocessing

This is one of the most important architectural decisions in Phase 5.

Preprocessing transformations that learn parameters from data are fitted **only on the training population**.

The workflow is:

```text
Train
    ↓
fit + transform

Validation
    ↓
transform only

Test
    ↓
transform only
```

For example, if a numerical imputer learns a median:

```text
Training data
    ↓
learn median
```

that median is then reused for:

```text
Validation
Test
Future inference
```

The validation and test distributions are never used to calculate preprocessing statistics.

### Why?

If preprocessing were fitted on the complete dataset before splitting, information from validation or test data could influence the representation used by the model.

This would create a form of data leakage and produce overly optimistic evaluation results.

---

# 14. Processed Feature Matrix

The finalized Phase 5 pipeline routes:

```text
81 raw features
```

through the preprocessing transformations.

Because preprocessing creates:

* missingness indicators;
* structural indicators;
* one-hot encoded categories;

the resulting representation expands to:

```text
241 processed features
```

The target is then appended separately:

```text
241 processed features
+
1 default target
=
242 columns
```

Final artifacts:

| Dataset    |    Rows | Columns |
| ---------- | ------: | ------: |
| Train      | 943,669 |     242 |
| Validation | 202,215 |     242 |
| Test       | 202,215 |     242 |

All three datasets contain:

```text
0 missing values
```

and preserve the:

```text
19.98% default rate
```

---

# 15. Persisted Artifacts

The preprocessing workflow produces canonical downstream artifacts.

```text
data/
└── processed/
    ├── train.parquet
    ├── validation.parquet
    └── test.parquet
```

The fitted preprocessing object is stored separately:

```text
models/
└── preprocessing/
    └── preprocessor.joblib
```

### Why save the processed datasets?

The original accepted-loans CSV is approximately 1.5 GB and requires CSV parsing before modeling.

Repeatedly rebuilding the processed matrix would unnecessarily repeat:

```text
CSV parsing
Target construction
Train/test splitting
Preprocessing
Categorical encoding
Missing-value handling
```

The canonical Parquet datasets therefore provide a stable modeling boundary.

Downstream model experiments can load:

```text
train.parquet
validation.parquet
test.parquet
```

directly.

### Why save the fitted preprocessor?

The preprocessing object contains the parameters learned from the training population.

Saving it ensures that the exact same transformations can later be applied to:

* validation data;
* test data;
* new inference data;
* deployed API requests.

This prevents training-time and inference-time preprocessing from diverging.

---

# 16. Why Parquet Instead of CSV?

The processed datasets are stored as Parquet rather than CSV.

This decision was made because the processed datasets are:

* predominantly numerical;
* considerably smaller than the original CSV representation;
* intended for repeated programmatic access;
* structured tabular data.

Parquet also preserves column types more reliably than CSV and is substantially more appropriate as a machine-learning data artifact.

The original CSV remains the source of truth.

The Parquet files are derived artifacts.

---

# 17. Notebook vs Production Code

The preprocessing notebook is retained for:

* validation;
* inspection;
* experimentation;
* documenting the reasoning behind preprocessing decisions;
* manually verifying intermediate results.

However, the notebook is **not a dependency of the production pipeline**.

The actual production workflow lives in:

```text
src/features/preprocessing.py
```

The production entry point is:

```python
prepare_preprocessed_data()
```

The intended execution path is therefore:

```text
Raw CSV
    ↓
src/features/preprocessing.py
    ↓
Processed artifacts
```

rather than:

```text
Notebook
    ↓
saved intermediate CSV
    ↓
production code
```

This prevents notebook state from becoming part of the system's execution logic.

---

# 18. Current Preprocessing Architecture

The finalized Phase 5 architecture is:

```text
                    RAW DATA
                       │
                       ▼
        accepted_2007_to_2018Q4.csv
                       │
                       ▼
                DatasetLoader
                       │
                       ▼
                TargetBuilder
                       │
                       ▼
             Observed Outcomes
                       │
                       ▼
              70/15/15 Split
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
          TRAIN ONLY        VALID / TEST
              │                 │
              ▼                 │
       Fit Preprocessor          │
              │                 │
              └────────┬────────┘
                       ▼
               Transform All Sets
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
        Train       Validation     Test
       Parquet       Parquet      Parquet
          │            │            │
          └────────────┼────────────┘
                       │
                       ▼
             preprocessor.joblib
```

---

# 19. Validation and Testing

The custom preprocessing transformers have dedicated unit tests.

Current test coverage includes:

```text
test_missing_indicator_imputer
test_event_history_imputer
test_structural_missingness_imputer
test_missing_indicator_feature_names
```

All tests currently pass:

```text
4 passed
```

The full test suite also passes:

```text
4 passed
```

The production preprocessing workflow was additionally validated end-to-end from a fresh notebook kernel using the original CSV.

Final artifact validation confirmed:

```text
Train:
943,669 rows
242 columns
19.98% default
0 missing values

Validation:
202,215 rows
242 columns
19.98% default
0 missing values

Test:
202,215 rows
242 columns
19.98% default
0 missing values
```

---

# 20. Important Decisions and Rationale

The following decisions are considered finalized for Phase 5.

| Decision                                            | Rationale                                                                     |
| --------------------------------------------------- | ----------------------------------------------------------------------------- |
| Use original accepted-loans CSV as production input | Keeps raw data as the source of truth                                         |
| Do not depend on EDA Parquet                        | EDA artifacts are analysis-specific                                           |
| Construct target before modeling                    | Establishes the correct observed-outcome population                           |
| Exclude ongoing outcomes                            | Their final outcome is not yet observed                                       |
| Use 70/15/15 split                                  | Provides training, model-selection, and unbiased final evaluation populations |
| Stratify on default                                 | Preserves the class distribution                                              |
| Fit preprocessing only on train                     | Prevents preprocessing leakage                                                |
| Preserve missingness indicators                     | Missingness can itself contain predictive information                         |
| Treat event-history missingness separately          | Missing event may mean no event occurred                                      |
| Preserve structural missingness                     | Some missingness is caused by application structure                           |
| Explicit categorical missing category               | Prevents loss of missingness information                                      |
| `handle_unknown="ignore"`                           | Makes inference robust to unseen categories                                   |
| Defer high-cardinality features                     | Their treatment requires feature-engineering decisions                        |
| Defer `earliest_cr_line`                            | Date-to-credit-history representation belongs to feature engineering          |
| Save Parquet artifacts                              | Faster and more appropriate downstream data access                            |
| Save fitted preprocessor                            | Guarantees consistent future transformations                                  |
| Keep notebook as validation artifact                | Separates experimentation from production execution                           |

---

# 21. What Phase 5 Does Not Do

The following are intentionally **not** part of Phase 5:

### High-cardinality encoding

```text
emp_title
title
zip_code
```

These will be considered during Feature Engineering.

### Date-derived features

```text
earliest_cr_line
```

will later be considered for representations such as credit-history duration.

### New business features

Examples such as:

```text
credit utilization ratios
income-to-loan ratios
payment burden
credit age
delinquency severity
```

belong to Feature Engineering.

### Model-specific transformations

Model-specific scaling, interaction terms, selection, and optimization decisions are not finalized during this phase.

---

# 22. Phase 5 Completion Criteria

Phase 5 is considered complete when:

* [x] Raw CSV is the production input.
* [x] Target construction is implemented.
* [x] Observed-outcome population is defined.
* [x] Train/validation/test split is implemented.
* [x] Stratification is applied.
* [x] Numerical missingness handling is implemented.
* [x] Event-history missingness handling is implemented.
* [x] Structural missingness handling is implemented.
* [x] Categorical missingness handling is implemented.
* [x] One-hot encoding is implemented.
* [x] Preprocessor is fitted only on training data.
* [x] 81 Phase-5 features are routed.
* [x] 241 processed features are produced.
* [x] No missing values remain in processed datasets.
* [x] Train/validation/test feature dimensions match.
* [x] Canonical Parquet datasets are saved.
* [x] Fitted preprocessing object is saved.
* [x] Unit tests pass.
* [x] End-to-end workflow passes from a fresh execution context.

---

# 23. Transition to Phase 6

Phase 5 establishes a clean and reproducible model-ready baseline.

The next phase is **Feature Engineering**.

The four explicitly deferred features provide the first feature-engineering candidates:

```text
emp_title
title
zip_code
earliest_cr_line
```

Feature Engineering will also investigate whether domain-derived variables can improve the representation of borrower credit risk without introducing target leakage.

The key principle carried forward is:

> Create predictive representations using only information that would genuinely be available at the point in time at which the model is intended to make its prediction.
