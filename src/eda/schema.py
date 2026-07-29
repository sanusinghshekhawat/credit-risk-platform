from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class TargetSummary:
    target_column: str

    total_samples: int
    missing_values: int

    n_classes: int

    unique_labels: list[str]

    class_counts: dict[str, int]
    class_percentages: dict[str, float]


@dataclass(slots=True, frozen=True)
class MissingSummary:
    total_columns: int
    columns_with_missing: int

    missing_counts: dict[str, int]
    missing_percentages: dict[str, float]


@dataclass(slots=True, frozen=True)
class NumericSummary:
    constant_columns: list[str]
    near_constant_columns: list[str]
    infinite_value_columns: list[str]

    skewness: dict[str, float]
    kurtosis: dict[str, float]


@dataclass(slots=True, frozen=True)
class CategoricalSummary:
    cardinality: dict[str, int]

    dominant_categories: dict[str, str]

    rare_categories: dict[str, list[str]]


@dataclass(slots=True, frozen=True)
class CorrelationSummary:
    high_correlation_pairs: list[tuple[str, str, float]]

    perfectly_correlated_pairs: list[tuple[str, str]]


@dataclass(slots=True, frozen=True)
class OutlierSummary:
    iqr_outlier_counts: dict[str, int]

    outlier_percentages: dict[str, float]


@dataclass(slots=True, frozen=True)
class BivariateSummary:
    numeric_vs_target: dict[str, float]

    categorical_vs_target: dict[str, float]


@dataclass(slots=True, frozen=True)
class FeatureReviewSummary:
    constant_columns: list[str]

    high_missing_columns: list[str]

    highly_correlated_columns: list[str]

    leakage_columns: list[str]

    review_columns: list[str]

    recommended_drop_columns: list[str]

    recommended_keep_columns: list[str]


@dataclass(slots=True, frozen=True)
class EDAReport:
    target: TargetSummary
    missing: MissingSummary
    numeric: NumericSummary
    categorical: CategoricalSummary
    correlation: CorrelationSummary
    outliers: OutlierSummary
