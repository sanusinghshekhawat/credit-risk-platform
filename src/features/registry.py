"""
Define the finalized feature registry for the LendingClub credit-risk model.

This module stores the dataset-specific feature decisions established during
data understanding and exploratory analysis.

It defines which features are retained, dropped, or used as targets and
groups retained features according to their required preprocessing treatment.

Transformation logic is intentionally kept separate in transformers.py.
"""

from __future__ import annotations

from src.features.schema import (
    FeatureDefinition,
    FeatureStatus,
    FeatureType,
    MissingStrategy,
)

STANDARD_NUMERICAL_FEATURES = (
    "loan_amnt",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "mths_since_recent_bc",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "collections_12_mths_ex_med",
    "acc_open_past_24mths",
    "acc_now_delinq",
    "tot_coll_amt",
    "tot_cur_bal",
    "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_rev_tl_op",
    "mo_sin_rcnt_tl",
    "mort_acc",
    "num_accts_ever_120_pd",
    "num_actv_bc_tl",
    "num_actv_rev_tl",
    "num_bc_sats",
    "num_bc_tl",
    "num_il_tl",
    "num_op_rev_tl",
    "num_rev_accts",
    "num_rev_tl_bal_gt_0",
    "num_sats",
    "num_tl_120dpd_2m",
    "num_tl_30dpd",
    "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m",
    "pct_tl_nvr_dlq",
    "percent_bc_gt_75",
    "pub_rec_bankruptcies",
    "tax_liens",
    "tot_hi_cred_lim",
    "total_bal_ex_mort",
    "total_bc_limit",
    "total_il_high_credit_limit",
    "avg_cur_bal",
    "bc_open_to_buy",
    "bc_util",
    "chargeoff_within_12_mths",
    "delinq_amnt",
    "total_rev_hi_lim",
    "il_util",
    "all_util",
)


EVENT_HISTORY_FEATURES = (
    "mths_since_last_delinq",
    "mths_since_last_record",
    "mths_since_last_major_derog",
    "mths_since_recent_bc_dlq",
    "mths_since_recent_revol_delinq",
    "mths_since_rcnt_il",
    "mths_since_recent_inq",
)


STRUCTURAL_NUMERICAL_FEATURES = (
    "open_acc_6m",
    "open_act_il",
    "open_il_12m",
    "open_il_24m",
    "total_bal_il",
    "open_rv_12m",
    "open_rv_24m",
    "max_bal_bc",
    "inq_fi",
    "total_cu_tl",
    "inq_last_12m",
)


JOINT_FEATURES = (
    "annual_inc_joint",
    "dti_joint",
    "verification_status_joint",
)


CATEGORICAL_FEATURES = (
    "term",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
    "application_type",
)


HIGH_CARDINALITY_FEATURES = (
    "emp_title",
    "title",
    "zip_code",
)


DATETIME_FEATURES = ("earliest_cr_line",)


DROPPED_FEATURES = (
    "funded_amnt",
    "funded_amnt_inv",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "issue_d",
    "initial_list_status",
)


TARGET_FEATURES = (
    "loan_status",
    "default",
)


def _build_feature_definitions() -> tuple[FeatureDefinition, ...]:
    """Build the complete feature registry."""

    definitions: list[FeatureDefinition] = []

    for feature in STANDARD_NUMERICAL_FEATURES:
        definitions.append(
            FeatureDefinition(
                name=feature,
                feature_type=FeatureType.NUMERICAL,
                status=FeatureStatus.KEEP,
                missing_strategy=MissingStrategy.MEDIAN,
            )
        )

    for feature in EVENT_HISTORY_FEATURES:
        definitions.append(
            FeatureDefinition(
                name=feature,
                feature_type=FeatureType.NUMERICAL,
                status=FeatureStatus.KEEP,
                missing_strategy=MissingStrategy.SPECIAL,
            )
        )

    for feature in STRUCTURAL_NUMERICAL_FEATURES:
        definitions.append(
            FeatureDefinition(
                name=feature,
                feature_type=FeatureType.NUMERICAL,
                status=FeatureStatus.KEEP,
                missing_strategy=MissingStrategy.SPECIAL,
            )
        )

    for feature in JOINT_FEATURES:
        definitions.append(
            FeatureDefinition(
                name=feature,
                feature_type=(
                    FeatureType.CATEGORICAL
                    if feature == "verification_status_joint"
                    else FeatureType.NUMERICAL
                ),
                status=FeatureStatus.KEEP,
                missing_strategy=MissingStrategy.SPECIAL,
            )
        )

    for feature in CATEGORICAL_FEATURES:
        definitions.append(
            FeatureDefinition(
                name=feature,
                feature_type=FeatureType.CATEGORICAL,
                status=FeatureStatus.KEEP,
                missing_strategy=MissingStrategy.MISSING_CATEGORY,
            )
        )

    for feature in HIGH_CARDINALITY_FEATURES:
        definitions.append(
            FeatureDefinition(
                name=feature,
                feature_type=FeatureType.CATEGORICAL,
                status=FeatureStatus.KEEP,
                missing_strategy=MissingStrategy.MISSING_CATEGORY,
            )
        )

    for feature in DATETIME_FEATURES:
        definitions.append(
            FeatureDefinition(
                name=feature,
                feature_type=FeatureType.DATETIME,
                status=FeatureStatus.KEEP,
                missing_strategy=MissingStrategy.NONE,
            )
        )

    for feature in DROPPED_FEATURES:
        definitions.append(
            FeatureDefinition(
                name=feature,
                feature_type=FeatureType.NUMERICAL,
                status=FeatureStatus.DROP,
            )
        )

    for feature in TARGET_FEATURES:
        definitions.append(
            FeatureDefinition(
                name=feature,
                feature_type=FeatureType.NUMERICAL,
                status=FeatureStatus.TARGET,
            )
        )

    return tuple(definitions)


FEATURE_REGISTRY = _build_feature_definitions()
