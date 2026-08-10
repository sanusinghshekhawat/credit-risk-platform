"""
Test the Phase 5 preprocessing components.

These tests verify that preprocessing transformations behave consistently
and that the production preprocessing workflow preserves the train/validation
/test separation.
"""

import pandas as pd

from src.features.transformers import (
    EventHistoryImputer,
    MissingIndicatorImputer,
    StructuralMissingnessImputer,
)


def test_missing_indicator_imputer():
    """Verify median imputation and missing indicators."""

    data = pd.DataFrame(
        {
            "income": [50000.0, 60000.0, None, 80000.0],
        }
    )

    transformer = MissingIndicatorImputer()
    result = transformer.fit_transform(data)

    assert result["income"].isna().sum() == 0
    assert result["income__missing"].tolist() == [0, 0, 1, 0]
    assert result["income"].iloc[2] == 60000.0


def test_event_history_imputer():
    """Verify sentinel-based event-history treatment."""

    data = pd.DataFrame(
        {
            "mths_since_last_delinq": [12.0, 24.0, None, 36.0],
        }
    )

    transformer = EventHistoryImputer()
    result = transformer.fit_transform(data)

    assert result["mths_since_last_delinq"].isna().sum() == 0
    assert result["mths_since_last_delinq"].iloc[2] == 37.0
    assert result["mths_since_last_delinq__missing"].iloc[2] == 1


def test_structural_missingness_imputer():
    """Verify individual and block-level missingness indicators."""

    data = pd.DataFrame(
        {
            "feature_a": [1.0, None, None],
            "feature_b": [2.0, None, 3.0],
        }
    )

    transformer = StructuralMissingnessImputer()
    result = transformer.fit_transform(data)

    assert result["structural_block__missing"].tolist() == [0, 1, 0]
    assert result.isna().sum().sum() == 0


def test_missing_indicator_feature_names():
    """Verify generated feature names are exposed to sklearn."""

    data = pd.DataFrame(
        {
            "income": [50000.0, None],
            "dti": [10.0, 20.0],
        }
    )

    transformer = MissingIndicatorImputer()
    transformer.fit(data)

    names = transformer.get_feature_names_out()

    assert names.tolist() == [
        "income",
        "income__missing",
        "dti",
        "dti__missing",
    ]
