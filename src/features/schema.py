"""
Define feature metadata used by the credit-risk preprocessing pipeline.

This module describes the semantic type, modeling status, and missing-value
strategy associated with each feature. It contains metadata only and does not
perform data transformations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FeatureType(StrEnum):
    """Supported semantic feature types."""

    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"


class MissingStrategy(StrEnum):
    """Supported missing-value handling strategies."""

    NONE = "none"
    MEDIAN = "median"
    MISSING_CATEGORY = "missing_category"
    SPECIAL = "special"


class FeatureStatus(StrEnum):
    """Modeling eligibility status of a feature."""

    KEEP = "keep"
    DROP = "drop"
    TARGET = "target"


@dataclass(frozen=True)
class FeatureDefinition:
    """Describe the preprocessing metadata for one feature."""

    name: str
    feature_type: FeatureType
    status: FeatureStatus
    missing_strategy: MissingStrategy = MissingStrategy.NONE
