# Exploratory Data Analysis Findings

## 1. Purpose

This document records the key findings and modeling-relevant conclusions from
the exploratory data analysis of the LendingClub loan dataset.

The objective of EDA was to understand the dataset structure, target outcome,
data quality, feature behavior, and relationships between application-time
characteristics and observed loan default.

EDA findings are used to guide preprocessing, feature engineering, and modeling.
They do not by themselves establish causal relationships or determine the final
model specification.

---

## 2. EDA Scope

The analysis covered:

* Dataset and feature structure
* Missing-value patterns
* Target distribution and outcome definition
* Feature eligibility and leakage considerations
* Numerical and categorical feature behavior
* Feature–target relationships
* Representative credit-risk factors
* Business and modeling implications

The targeted feature analysis focused on 18 representative feature concepts
covering loan characteristics, borrower characteristics, credit quality,
credit behavior, and application structure.

---

## 3. Target and Outcome Definition

The dataset contains multiple loan-status categories, including:

* Fully Paid
* Charged Off
* Current
* Late (16–30 days)
* Late (31–120 days)
* In Grace Period
* Default
* Other historical credit-policy statuses

A key EDA finding was that unresolved loans cannot automatically be treated as
non-default observations.

Current, grace-period, and late loans may not yet have reached a finalized
outcome. Therefore, the modeling population must distinguish finalized observed
outcomes from unresolved loans.

The target construction is handled separately from the EDA-specific target
analysis so that the observed-outcome population can be reused consistently
during modeling.

---

## 4. Major Target Findings

The observed finalized outcome population is substantially imbalanced, with
non-default outcomes considerably more common than observed defaults.

This class imbalance must be considered during model evaluation and threshold
selection.

Accuracy alone should therefore not be used as the primary measure of model
quality.

---

## 5. Feature-Level Findings

### 5.1 Loan Amount

Loan amount showed a nonlinear relationship with observed default risk.

Default rates increased across the lower and middle loan-amount ranges before
flattening and declining for the highest loan amount category.

**Business implication:** Loan size is associated with differences in observed
risk, but the relationship is not simply monotonic.

**Modeling implication:** `loan_amnt` should be retained. The raw numerical
representation should initially be preserved, with alternative representations
evaluated during feature engineering.

---

### 5.2 FICO Score

FICO score showed a strong negative relationship with observed default risk.

Lower FICO bands exhibited substantially higher default rates, while default
rates declined progressively as FICO increased.

**Business implication:** Credit score is a strong indicator of borrower credit
quality and is highly relevant to credit-risk assessment.

**Modeling implication:** FICO should be retained as a core predictive feature.
Alternative representations such as bands may be evaluated later, but the raw
numerical representation should not be discarded based solely on EDA.

---

### 5.3 Interest Rate

Interest rate exhibited a strong and approximately monotonic positive
relationship with observed default risk.

Default rates increased substantially as interest rate increased.

**Business implication:** Higher interest rates are associated with materially
higher observed default risk.

**Modeling implication:** `int_rate` should be retained as a candidate
predictive feature. However, its inclusion must be validated against the
model's prediction point because the rate is assigned during underwriting and
may not be available for an earlier-stage approval decision.

---

### 5.4 Debt-to-Income Ratio

DTI showed a generally positive relationship with observed default risk.

Default rates increased from approximately 15% at low DTI levels to more than
30% in several high-DTI bands, although the extreme upper tail was less stable.

**Business implication:** Higher debt burden is associated with greater observed
default risk.

**Modeling implication:** `dti` should be retained as a candidate predictive
feature. Its raw numerical representation should initially be preserved, with
transformations or bands evaluated during feature engineering if required.

---

### 5.5 Delinquency History

`delinq_2yrs` showed a generally positive relationship with observed default
risk.

Borrowers with more recent delinquencies tended to exhibit higher default
rates, although the relationship became unstable in very sparse high-count
categories.

**Business implication:** Previous delinquency behavior provides information
about a borrower's repayment history and credit risk.

**Modeling implication:** `delinq_2yrs` should be retained. Sparse upper-tail
values should be handled carefully during preprocessing or feature engineering.

---

### 5.6 Annual Income

Default rate decreased consistently as annual income increased.

The lowest income band had a default rate of 24.08%, compared with 14.61% in
the highest-income band.

The upper income category covered a very wide range because annual income is
strongly right-skewed.

**Business implication:** Higher reported income is associated with lower
observed default risk and provides information about repayment capacity.

**Modeling implication:** `annual_inc` should be retained. The raw numerical
representation should initially be preserved, while transformations such as
log scaling can be evaluated later.

---

### 5.7 Revolving Credit Utilization

Revolving utilization showed a clear positive relationship with default risk.

Default rates increased from 14.68% below 10% utilization to 23.14% at
90–100% utilization. The 100%+ group had a higher rate of 26.85%, but was much
smaller.

**Business implication:** High revolving credit utilization may indicate
greater financial pressure and is associated with increased observed default
risk.

**Modeling implication:** `revol_util` should be retained. Its raw numerical
representation should initially be preserved.

---

### 5.8 Recent Credit Inquiries

`inq_last_6mths` showed a clear positive relationship with default risk.

Default rates increased from 17.92% for borrowers with no recent inquiries to
approximately 28% for borrowers with five or more inquiries.

**Business implication:** Increased recent credit-seeking activity is
associated with higher observed default risk.

**Modeling implication:** The feature should be retained. The sparse upper tail
should be handled carefully during feature engineering.

---

### 5.9 Total Credit Accounts

`total_acc` showed a relatively weak relationship with default risk.

Default rates declined modestly across the lower and middle account-count
ranges but became less clearly monotonic at higher values.

**Business implication:** Credit-history depth provides some contextual
information, but total account count alone is a relatively weak risk
differentiator.

**Modeling implication:** `total_acc` may be retained as a candidate feature,
but it should not receive special transformation priority based on EDA alone.

---

### 5.10 Loan Term

Loan term showed a strong categorical relationship with observed default risk.

* 36-month loans: 16.02% default rate
* 60-month loans: 32.45% default rate

**Business implication:** Longer-term loans are associated with substantially
higher observed default risk.

**Modeling implication:** `term` should be retained as a categorical feature
and represented explicitly during preprocessing.

---

### 5.11 Loan Purpose

Default rates varied substantially across loan purposes.

`small_business` had the highest observed default rate at 29.86%, while
`wedding` had the lowest at 12.43%.

Some categories were relatively small and therefore require cautious
interpretation.

**Business implication:** Loan purpose captures contextual differences between
borrower and loan populations and may provide useful risk information.

**Modeling implication:** `purpose` should be retained as a categorical
candidate feature. Rare categories should be handled carefully during
preprocessing.

---

### 5.12 Home Ownership

Among the major categories:

* RENT: 23.23%
* OWN: 20.63%
* MORTGAGE: 17.23%

The `OTHER`, `ANY`, and `NONE` groups were extremely small.

**Business implication:** Housing status is associated with differences in
observed default risk, potentially reflecting broader financial circumstances.

**Modeling implication:** `home_ownership` should be retained. Rare categories
require appropriate preprocessing.

---

### 5.13 Verification Status

Verification status showed a clear difference in observed default rates:

* Verified: 23.86%
* Source Verified: 20.96%
* Not Verified: 14.74%

The relationship should not be interpreted causally.

**Business implication:** Verification status may capture differences in borrower
profiles, financial characteristics, or underwriting practices.

**Modeling implication:** `verification_status` should be retained as a
candidate categorical feature and evaluated alongside income, DTI, and credit
quality.

---

### 5.14 Employment Length

Reported employment length showed only a weak relationship with default among
borrowers with known employment length.

However, the missing employment-length group had a substantially higher default
rate of 26.93%.

**Business implication:** Missing employment information itself may carry
information about borrower or application characteristics.

**Modeling implication:** `emp_length` should be retained, but missingness
should be handled explicitly rather than automatically replacing missing values
without preserving the missing-information signal.

---

### 5.15 Application Type

Joint applications had a higher observed default rate than individual
applications:

* Individual: 19.89%
* Joint App: 24.61%

Joint applications were substantially less numerous.

**Business implication:** Application structure may capture differences in
borrower or household financial circumstances.

**Modeling implication:** `application_type` should be retained as a candidate
categorical feature and evaluated alongside other borrower characteristics.

---

### 5.16 Public Derogatory Records

Borrowers with no public derogatory records had a 19.41% default rate, compared
with approximately 22–24% among borrowers with one or more records.

The relationship became relatively flat beyond the first few records.

**Business implication:** The presence of adverse public credit records is
associated with elevated observed default risk.

**Modeling implication:** `pub_rec` should be retained. Alternative
representations, including presence/absence indicators, can be evaluated
during feature engineering.

---

### 5.17 Recent Collections

Recent collection activity showed a positive relationship with default risk.

* 0 collections: 19.89%
* 1 collection: 25.39%
* 2 collections: 27.40%
* 3+ collections: 22.19%

The 3+ group was very small and therefore unstable.

**Business implication:** Recent collection activity is associated with prior
financial difficulty and higher observed default risk.

**Modeling implication:** `collections_12_mths_ex_med` should be retained.
A binary indicator for the presence of collections may be evaluated later
because the presence of collections appears more stable than the sparse upper
tail.

---

## 6. Cross-Feature EDA Conclusions

Several broad patterns emerged across the targeted analysis:

### Strong risk signals

The clearest relationships were observed for:

* FICO
* Interest rate
* DTI
* Loan term
* Revolving utilization
* Recent credit inquiries
* Annual income

### Credit-history signals

The following provided additional evidence of borrower credit behavior:

* Delinquencies
* Public derogatory records
* Recent collections
* Total credit accounts

### Contextual/application signals

The following showed meaningful category-level differences:

* Loan purpose
* Home ownership
* Verification status
* Employment length
* Application type

### Important caveat

Observed relationships represent associations with the observed outcome
population. They should not be interpreted as causal effects.

A feature can show a strong relationship with default because it is correlated
with other borrower characteristics or because it reflects underwriting
decisions.

---

## 7. Prediction-Time and Leakage Considerations

EDA findings do not override the project's feature eligibility framework.

Features must be evaluated according to whether they are legitimately available
at the chosen prediction point.

In particular, `int_rate` demonstrated a strong predictive relationship but
requires explicit review because the rate is assigned during underwriting.

Similarly, features representing events occurring after loan origination or
during repayment must not be used merely because they show strong relationships
with the final outcome.

Leakage analysis and feature eligibility rules remain authoritative for the
final modeling population.

---

## 8. EDA → Modeling Decision Summary

| Feature                      | EDA finding                                | Initial decision        |
| ---------------------------- | ------------------------------------------ | ----------------------- |
| `loan_amnt`                  | Nonlinear relationship                     | Retain                  |
| FICO                         | Strong negative relationship               | Retain                  |
| `int_rate`                   | Strong positive relationship               | Prediction-point review |
| `dti`                        | Positive relationship                      | Retain                  |
| `delinq_2yrs`                | Generally positive                         | Retain                  |
| `annual_inc`                 | Negative relationship                      | Retain                  |
| `revol_util`                 | Positive relationship                      | Retain                  |
| `inq_last_6mths`             | Positive relationship                      | Retain                  |
| `total_acc`                  | Weak relationship                          | Retain; low priority    |
| `term`                       | Strong categorical difference              | Retain                  |
| `purpose`                    | Meaningful categorical differences         | Retain                  |
| `home_ownership`             | Meaningful categorical differences         | Retain                  |
| `verification_status`        | Meaningful categorical differences         | Retain                  |
| `emp_length`                 | Weak relationship; missingness informative | Retain                  |
| `application_type`           | Meaningful categorical difference          | Retain                  |
| `pub_rec`                    | Weak positive relationship                 | Retain                  |
| `collections_12_mths_ex_med` | Positive; sparse upper tail                | Retain                  |

---

## 9. Transition to Preprocessing

EDA is considered complete once the documented findings and feature-eligibility
decisions have been reviewed.

The next stage is preprocessing.

Preprocessing will address:

* datatype normalization
* datetime handling
* missing-value treatment
* categorical encoding
* numerical transformations
* rare-category handling
* feature representation
* final model-ready dataset construction

EDA findings should guide these decisions, but preprocessing should remain
separate from exploratory analysis to prevent exploratory transformations from
being confused with the final modeling pipeline.
