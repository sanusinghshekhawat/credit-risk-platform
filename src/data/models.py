from dataclasses import dataclass
from enum import Enum

import pandas as pd


@dataclass(
    slots=True,
    frozen=True,
)
class DataProfile:
    """
    Immutable summary of a dataset.
    """

    rows: int
    columns: int

    memory_mb: float

    duplicate_rows: int

    missing_values: dict[str, int]

    numerical_columns: list[str]

    categorical_columns: list[str]

    datetime_columns: list[str]

    boolean_columns: list[str]

    descriptive_statistics: pd.DataFrame


@dataclass(slots=True, frozen=True)
class DataDictionaryEntry:
    """
    Metadata describing a single dataset column.
    """

    column: str
    dtype: str
    missing_count: int
    missing_percentage: float
    unique_count: int


class LeakageDecision(str, Enum):
    KEEP = "KEEP"
    REVIEW = "REVIEW"
    DROP = "DROP"
    LEAKAGE = "LEAKAGE"
    TARGET = "TARGET"
    UNKNOWN = "UNKNOWN"


@dataclass(slots=True, frozen=True)
class LeakageRule:
    """Leakage classification for a single feature."""

    column: str
    decision: LeakageDecision
    reason: str
    configured: bool


@dataclass(slots=True, frozen=True)
class LeakageReport:
    """Summary of leakage decisions across all dataset columns."""

    rules: list[LeakageRule]

    keep: list[str]
    review: list[str]
    drop: list[str]
    leakage: list[str]
    target: list[str]
    unknown: list[str]
