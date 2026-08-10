# Credit Risk Platform — Project Architecture

## 1. Purpose

This document defines the architectural boundaries, responsibilities, and
workflow conventions for the Credit Risk Platform.

The purpose is to ensure that reusable logic, exploratory analysis,
configuration, data artifacts, tests, documentation, modeling, deployment,
and monitoring remain clearly separated throughout the project.

This document is an architectural reference. New code and project artifacts
should follow these boundaries unless there is a deliberate architectural
reason to change them.

---

## 2. High-Level Architecture

The project follows a staged machine-learning workflow:

```text
Raw Data
   │
   ▼
Data Understanding
   │
   ▼
Target & Leakage Definition
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Preprocessing
   │
   ▼
Feature Engineering
   │
   ▼
Train / Validation / Test
   │
   ▼
Model Development
   │
   ▼
Model Evaluation
   │
   ▼
Deployment
   │
   ▼
Monitoring
```

The implementation is divided into reusable source modules, data artifacts,
tests, documentation, and application/deployment components.

---

## 3. Repository Structure

```text
credit-risk-platform/
│
├── .github/
│   └── workflows/
│
├── app/
│
├── configs/
│   └── leakage_rules.yaml
│
├── data/
│   ├── external/
│   ├── interim/
│   │   └── eda_dataset.parquet
│   ├── processed/
│   └── raw/
│       ├── archives/
│       └── extracted/
│
├── docs/
│   ├── architecture/
│   ├── data/
│   ├── deployment/
│   ├── eda/
│   └── modeling/
│
├── logs/
│
├── models/
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   └── 2_data_understanding.ipynb
│
├── reports/
│   ├── figures/
│   └── tables/
│
├── scripts/
│
├── src/
│   ├── api/
│   ├── config/
│   ├── data/
│   ├── eda/
│   ├── features/
│   ├── models/
│   ├── monitoring/
│   ├── pipelines/
│   ├── utils/
│   └── visualization/
│
├── tests/
│
├── .gitignore
├── .pre-commit-config.yaml
├── environment.yml
├── LICENSE
├── Makefile
├── pyproject.toml
├── requirements.txt
└── ruff.toml
```

---

## 4. Architectural Principles

### 4.1 Notebooks are not the production implementation

Notebooks are primarily for:

* exploration
* visualization
* investigation
* lightweight validation
* demonstrating results

Reusable logic must live under `src/`.

A transformation that will be required during training and inference must not
exist only as notebook code.

---

### 4.2 Reusable logic belongs in `src/`

If a function or transformation could reasonably be reused during:

* preprocessing
* feature engineering
* train/test preparation
* model training
* validation
* inference

it should be implemented as reusable source code rather than duplicated in a
notebook.

---

### 4.3 Pipelines orchestrate; components transform

The project separates individual transformations from the orchestration of
those transformations.

```text
src/features/
        │
        │ individual reusable transformations
        ▼
src/pipelines/
        │
        │ orchestration
        ▼
model-ready dataset / model
```

`features/` contains feature-level transformation logic.

`pipelines/` combines those components into reproducible workflows.

---

## 5. `src/config/`

### Responsibility

Centralized configuration and project paths.

```text
src/config/
├── __init__.py
├── paths.py
└── settings.py
```

### `paths.py`

Responsible for project filesystem locations.

Examples include:

* raw data
* interim data
* processed data
* reports
* models
* logs

Code elsewhere in the project should use the centralized path definitions
rather than hardcoding project-relative paths.

### `settings.py`

Contains project/data configuration that should be centralized rather than
duplicated throughout the codebase.

---

## 6. `src/data/`

### Responsibility

Data ingestion, inspection, profiling, dictionary information, leakage
classification, and target-population construction.

```text
src/data/
├── dictionary.py
├── inspector.py
├── leakage.py
├── loader.py
├── models.py
├── profiler.py
└── target.py
```

### Important boundary

`src/data/` answers:

> What is the data and what population are we working with?

It should not become a general-purpose location for model preprocessing.

### `target.py`

Contains reusable target/population construction logic.

This is intentionally separate from:

```text
src/eda/target.py
```

because:

* `src/data/target.py` constructs the modeling target/population.
* `src/eda/target.py` analyzes the target for EDA.

---

## 7. `src/eda/`

### Responsibility

Exploratory analysis only.

```text
src/eda/
├── bivariate.py
├── categorical.py
├── config.py
├── correlation.py
├── engine.py
├── feature_review.py
├── missing.py
├── numeric.py
├── outliers.py
├── schema.py
└── target.py
```

EDA components answer:

> What does the data tell us?

They may calculate summaries, distributions, correlations, relationships,
outliers, and feature-level observations.

EDA should not silently become the implementation of the final preprocessing
pipeline.

---

## 8. `src/features/`

### Responsibility

Reusable feature transformations used to convert valid input variables into
model-ready representations.

Examples may eventually include:

* numerical transformations
* categorical transformations
* missing-value transformations
* rare-category handling
* domain-specific feature engineering
* derived credit-risk features

A feature transformation that must be identical during training and prediction
belongs here rather than being manually recreated in notebooks.

---

## 9. `src/pipelines/`

### Responsibility

Orchestrate reusable components into end-to-end workflows.

Examples:

```text
preprocessing pipeline
training pipeline
inference pipeline
```

The pipeline should define the sequence in which transformations are applied
and ensure that the same fitted transformations can be reused during inference.

---

## 10. `src/models/`

### Responsibility

Model-related implementation.

This area will eventually contain:

* model definitions/wrappers
* model training logic
* model configuration
* model persistence interfaces
* prediction logic where appropriate

Model code should consume prepared features rather than performing hidden,
inconsistent preprocessing internally.

---

## 11. `src/visualization/`

### Responsibility

Reusable visualization utilities.

Notebook-specific exploratory plots may remain in notebooks when they are
truly one-off.

Reusable plotting functionality should be implemented here.

---

## 12. `src/api/`

### Responsibility

API-facing functionality for serving the model or platform functionality.

This should not contain the core feature engineering or model-training logic.

The API should call the appropriate reusable pipeline/model components.

---

## 13. `src/monitoring/`

### Responsibility

Post-deployment monitoring.

Potential responsibilities include:

* prediction monitoring
* data drift
* feature drift
* model performance monitoring
* operational health
* alerting

Monitoring logic should consume production outputs rather than being mixed
into training code.

---

## 14. `src/utils/`

### Responsibility

General utilities that are genuinely cross-cutting and do not belong to a
specific domain.

Examples include:

* logging
* project utilities
* shared infrastructure helpers

Utilities should not become a dumping ground for feature-specific or
model-specific logic.

---

## 15. Data Artifact Boundaries

The project uses explicit data stages:

```text
data/raw/
      │
      ▼
data/interim/
      │
      ▼
data/processed/
```

### `data/raw/`

Original source data.

Raw data must remain unchanged.

### `data/interim/`

Intermediate datasets produced during data understanding and EDA.

Current example:

```text
data/interim/eda_dataset.parquet
```

### `data/processed/`

Final model-ready datasets or other outputs produced by preprocessing.

Preprocessing should write its validated output here rather than overwriting
the raw or interim dataset.

---

## 16. Notebook Policy

Current notebooks:

```text
notebooks/
├── 01_exploratory_data_analysis.ipynb
└── 2_data_understanding.ipynb
```

### EDA notebook

`01_exploratory_data_analysis.ipynb` is the exploratory analysis record.

It contains:

* EDA execution
* visualizations
* observations
* exploratory validation

### Data-understanding notebook

`2_data_understanding.ipynb` contains earlier data-understanding work and
supporting investigation.

### Future notebooks

A new notebook should only be created when it provides meaningful exploratory
or validation value.

We do not create one notebook per project phase merely for the sake of having
a notebook.

Production/reusable logic belongs in `src/`.

---

## 17. Testing Architecture

The `tests/` directory is the validation boundary for reusable source code.

Tests should eventually cover:

```text
tests/
├── data/
├── eda/
├── features/
├── pipelines/
└── models/
```

The exact structure may evolve as implementation grows.

### Testing principle

Notebook execution is not a substitute for automated tests.

Reusable transformations should have deterministic tests covering:

* expected output
* edge cases
* missing values
* invalid inputs
* train/inference consistency where applicable

---

## 18. Documentation Architecture

Documentation is separated by purpose:

```text
docs/
├── architecture/
├── data/
├── deployment/
├── eda/
└── modeling/
```

### `docs/architecture/`

How the system is structured and why.

### `docs/data/`

What the dataset contains and how data eligibility/leakage is defined.

Current documents:

```text
data_dictionary.md
leakage_analysis.md
```

### `docs/eda/`

What exploratory analysis discovered.

Current document:

```text
eda_findings.md
```

### `docs/modeling/`

Modeling decisions, assumptions, evaluation strategy, and related rationale.

### `docs/deployment/`

Deployment architecture and operational documentation.

---

## 19. Configuration vs Code

Configuration that may change independently from implementation should be
centralized.

For example:

```text
configs/leakage_rules.yaml
```

should contain configurable leakage/eligibility rules rather than embedding
all such decisions directly into Python code.

Python modules should consume these configurations.

---

## 20. Preprocessing Architecture

Preprocessing is the next major implementation stage.

The intended boundary is:

```text
data/interim/eda_dataset.parquet
                │
                ▼
       preprocessing components
          src/features/
                │
                ▼
       preprocessing pipeline
          src/pipelines/
                │
                ▼
data/processed/model_dataset.parquet
```

The preprocessing implementation must distinguish between:

### Deterministic transformations

Transformations that do not learn parameters from the dataset.

### Learned transformations

Transformations that estimate information from data, such as:

* imputation values
* scaling parameters
* learned category mappings
* quantile boundaries
* frequency-based rules

Learned transformations must be fitted using the training population only.

The fitted preprocessing pipeline must subsequently be reusable during
validation, testing, and prediction.

---

## 21. Prediction Consistency

A central architectural requirement is:

```text
TRAINING

raw valid features
       ↓
fit preprocessing
       ↓
transformed features
       ↓
model


PREDICTION

new valid features
       ↓
same fitted preprocessing
       ↓
transformed features
       ↓
same model
```

The preprocessing logic used during prediction must not be manually recreated
from the training notebook.

This prevents training/serving skew and inconsistent feature representations.

---

## 22. Phase Ownership

The project stages have distinct responsibilities:

| Stage               | Primary responsibility                             |
| ------------------- | -------------------------------------------------- |
| Data Understanding  | Understand structure and quality                   |
| Target Definition   | Define observed outcome population                 |
| Leakage Analysis    | Determine feature eligibility                      |
| EDA                 | Discover relationships and patterns                |
| Preprocessing       | Convert valid data into consistent representations |
| Feature Engineering | Create model-relevant derived features             |
| Modeling            | Train and compare models                           |
| Evaluation          | Assess predictive and business performance         |
| Deployment          | Serve the selected model                           |
| Monitoring          | Observe production behavior                        |

A later stage should not silently redefine decisions owned by an earlier stage.

For example, preprocessing should not silently redefine the target or override
leakage decisions.

---

## 23. Current Project State

At the beginning of preprocessing:

```text
Data understanding       COMPLETE
Target definition        COMPLETE
Leakage analysis         COMPLETE
EDA                      COMPLETE
EDA documentation       COMPLETE

Preprocessing            NEXT
Feature engineering      PENDING
Model development        PENDING
Evaluation               PENDING
Deployment               PENDING
Monitoring               PENDING
```

The current working branch is:

```text
feature/preprocessing
```

The preprocessing branch was created from the completed
`feature/data-understanding` branch.

---

## 24. Architectural Rule for Future Work

Before adding new code, ask:

> Is this exploratory, reusable transformation logic, orchestration,
> configuration, modeling, deployment, or monitoring?

Then place it in the corresponding architectural boundary.

In particular:

* Do not put reusable preprocessing logic in notebooks.
* Do not put feature transformations in `src/data/` merely because they operate
  on data.
* Do not put pipeline orchestration inside individual feature modules.
* Do not duplicate training transformations inside prediction code.
* Do not overwrite raw data.
* Do not use test/validation information to fit learned preprocessing steps.
* Do not introduce new project structure without a clear architectural reason.

This architecture should be treated as the default project convention for the
remaining development unless a deliberate design change is documented.
